"""
IdeaToEpic - Multi-agent requirements generation pipeline
Transforms VOC input into structured product backlogs using LangGraph
"""

import os
import json
import time
import logging
from typing import TypedDict, Literal, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("idea2epic")

# ─────────────────────────────────────────────
# LANGSMITH TRACING (optional, zero-code-change)
# Set these in your .env to enable:
#   LANGCHAIN_TRACING_V2=true
#   LANGCHAIN_API_KEY=your_langsmith_key
#   LANGCHAIN_PROJECT=idea2epic
# LangChain/LangGraph picks them up automatically.
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# 1. SETUP
# ─────────────────────────────────────────────

# Config 
MAX_ITERATIONS = 3 # 1 initial attempt + 2 revision cycles

# LLM setup - replace with your preferred model
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)


# State definition
class RequirementsState(TypedDict):
    # Input
    product_domain: str           # e.g. "hospital scheduling app"
    generate_voc: bool            # True = auto-generate VOC
    voc_input: str                # raw VOC text (user-written or generated)

    # Pipeline outputs
    stakeholder_needs: list       # extracted from VOC
    epics: list                   # high-level capabilities
    features: list                # mid-level features per epic
    user_stories: list            # stories with acceptance criteria

    # Quality control
    quality_feedback: str
    quality_score: Optional[int]  # 1-10 score from auditor
    quality_issues: list          # list of identified issues
    approved: bool
    # iteration counts completed quality checks (starts at 0).
    # Pipeline allows up to MAX_ITERATIONS checks before forcing exit.
    iteration: int

# ─────────────────────────────────────────────
# 2. SHARED PROMPT BUILDER
# ─────────────────────────────────────────────
def build_voc_prompt(product_domain: str) -> str:
    """Generate VOC creation prompt"""
    return f"""You are a product discovery specialist. Generate a realistic \
Voice of Customer (VOC) input for a {product_domain} product.

Include:
- 2-3 different user personas with distinct needs and roles
- Real pain points (specific, not generic)
- Desired outcomes, not just features
- Some conflicting needs between personas (this is realistic!)
- Written in natural conversational language, as if from stakeholder interviews

Do NOT write requirements or user stories. Write raw customer voice only.
Output plain text, no headers or formatting."""


def clean_json(text: str) -> str:
    """Remove markdown code fences from LLM responses."""
    return text.replace("```json", "").replace("```", "").strip()


# ─────────────────────────────────────────────
# 3. AGENT NODES
# ─────────────────────────────────────────────

def voc_generator_node(state: RequirementsState) -> RequirementsState:
    """
    Generates a realistic VOC if the user didn't provide one.
    Only runs when generate_voc = True.
    """
    log.info("[VOC Generator] Starting VOC generation for domain: '%s'", state["product_domain"])
    t0 = time.time()

    response = llm.invoke([HumanMessage(content=build_voc_prompt(state["product_domain"]))])

    log.info("[VOC Generator] Done in %.1fs", time.time() - t0)
    return {**state, "voc_input": response.content}


def voc_analyst_node(state: RequirementsState) -> RequirementsState:
    """
    Extracts structured stakeholder needs from raw VOC text.
    """
    log.info("[VOC Analyst] Extracting stakeholder needs")
    t0 = time.time()

    prompt = f"""You are a senior business analyst specializing in requirements engineering.

Analyze this Voice of Customer input and extract structured stakeholder needs:

VOC INPUT:
{state['voc_input']}

Extract and return a JSON array of stakeholder needs. Each item must have:
- "persona": who this need belongs to
- "pain_point": what problem they have
- "desired_outcome": what they want to achieve
- "priority": High / Medium / Low

Return ONLY valid JSON, no explanation, no markdown code blocks.
Example format:
[
  {{
    "persona": "Hospital Nurse",
    "pain_point": "Cannot see patient schedule updates in real time",
    "desired_outcome": "Instant visibility of schedule changes across shifts",
    "priority": "High"
  }}
]"""

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        needs = json.loads(clean_json(response.content))
        log.info("[VOC Analyst] Extracted %d stakeholder needs in %.1fs", len(needs), time.time() - t0)
    except json.JSONDecodeError:
        log.warning("[VOC Analyst] JSON parse failed — wrapping raw response as fallback")
        needs = [{"raw": response.content}]

    return {**state, "stakeholder_needs": needs}


def requirement_architect_node(state: RequirementsState) -> RequirementsState:
    """
    Builds the Epic → Feature → User Story hierarchy with full traceability.
    On revision cycles, quality_feedback is injected into the prompt.
    """
    iteration = state.get("iteration", 0)
    log.info("[Requirement Architect] Building backlog (attempt %d/%d)", iteration + 1, MAX_ITERATIONS)
    t0 = time.time()

    needs_text = json.dumps(state["stakeholder_needs"], indent=2)
    feedback_section = (
        f"Quality feedback to address:\n{state['quality_feedback']}"
        if state.get("quality_feedback")
        else "No prior feedback — this is the first attempt."
    )

    prompt = f"""You are a senior systems engineer specializing in requirements architecture.

Given these stakeholder needs, generate a fully traceable product backlog hierarchy.

STAKEHOLDER NEEDS:
{needs_text}

{feedback_section}

Generate the following structure as valid JSON:
{{
  "epics": [
    {{
      "id": "E1",
      "title": "Epic title",
      "description": "What capability this epic delivers",
      "features": [
        {{
          "id": "F1.1",
          "epic_id": "E1",
          "title": "Feature title",
          "description": "What this feature enables",
          "user_stories": [
            {{
              "id": "US1.1.1",
              "feature_id": "F1.1",
              "story": "As a [persona], I want [goal] so that [benefit]",
              "acceptance_criteria": [
                "Given [context], When [action], Then [outcome]"
              ]
            }}
          ]
        }}
      ]
    }}
  ]
}}

Rules:
- 3-5 Epics
- 2-4 Features per Epic
- 2-3 User Stories per Feature
- Every ID must create a clear traceability chain (US references F, F references E)
- Acceptance criteria must be measurable and testable — no vague language
- Return ONLY valid JSON, no markdown, no explanation"""

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        hierarchy = json.loads(clean_json(response.content))
        epics = hierarchy.get("epics", [])
    except json.JSONDecodeError:
        log.warning("[Requirement Architect] JSON parse failed — storing raw response")
        epics = [{"raw": response.content}]

    # Flatten for easy access
    features = [f for e in epics for f in e.get("features", [])]
    stories = [s for f in features for s in f.get("user_stories", [])]
    
    log.info(
        "[Requirement Architect] Built %d epics, %d features, %d stories in %.1fs",
        len(epics), len(features), len(stories), time.time() - t0
    )
    return {**state, "epics": epics, "features": features, "user_stories": stories}


def quality_checker_node(state: RequirementsState) -> RequirementsState:
    """
    Reviews the full backlog for quality, gaps, and traceability.
    JSON parse failure → REJECTED (safe default — never silently approve broken output).
    """
    log.info("[Quality Checker] Running quality audit (check #%d)", state.get("iteration", 0) + 1)
    t0 = time.time()

    backlog_summary = {
        "epics_count": len(state["epics"]),
        "features_count": len(state["features"]),
        "stories_count": len(state["user_stories"]),
        "epics": state["epics"]
    }

    prompt = f"""You are a requirements quality auditor with 15+ years of experience.

Review this product backlog against the original stakeholder needs and identify issues.

ORIGINAL STAKEHOLDER NEEDS:
{json.dumps(state['stakeholder_needs'], indent=2)}

GENERATED BACKLOG:
{json.dumps(backlog_summary, indent=2)}

Check for:
1. Missing traceability links (stories without features, features without epics)
2. Ambiguous or untestable user stories
3. Stakeholder needs not covered by any epic/feature
4. Acceptance criteria that are vague or unmeasurable
5. Conflicting requirements not addressed

Return JSON:
{{
  "status": "APPROVED" or "REJECTED",
  "score": <integer 1-10>,
  "issues": ["issue 1", "issue 2"],
  "feedback": "Specific, actionable instructions for the architect to fix if REJECTED. Empty string if APPROVED."
}}

Return ONLY valid JSON, no markdown."""

    response = llm.invoke([HumanMessage(content=prompt)])
    new_iteration = state.get("iteration", 0) + 1

    try:
        result = json.loads(clean_json(response.content))
        approved = result.get("status") == "APPROVED"
        feedback = result.get("feedback", "")
        score = result.get("score")
        issues = result.get("issues", [])
        log.info(
            "[Quality Checker] Status: %s | Score: %s/10 | Issues: %d | Iteration: %d",
            result.get("status"), score, len(issues), new_iteration
        )
    except json.JSONDecodeError:
        # Safe default: reject so the architect gets another attempt
        log.warning("[Quality Checker] JSON parse failed — defaulting to REJECTED for safety")
        approved = False
        feedback = "Quality check response could not be parsed. Please regenerate the backlog with stricter JSON formatting."
        score = None
        issues = ["Quality auditor response was malformed"]

    log.info("[Quality Checker] Done in %.1fs", time.time() - t0)

    return {
        **state,
        "approved": approved,
        "quality_feedback": feedback,
        "quality_score": score,
        "quality_issues": issues,
        "iteration": new_iteration,
    }


# ─────────────────────────────────────────────
# 4. ROUTING LOGIC
# ─────────────────────────────────────────────

def should_generate_voc(state: RequirementsState) -> Literal["voc_generator", "voc_analyst"]:
    """Route to VOC generator or skip straight to analyst."""
    if state.get("generate_voc", False):
        return "voc_generator"
    return "voc_analyst"


def quality_gate(state: RequirementsState) -> Literal["requirement_architect", "end"]:
    """
    Loop back to architect if quality fails.
    Exits when: approved=True OR iteration >= MAX_ITERATIONS.
    MAX_ITERATIONS = 3 means: 1 original + 2 revision cycles.
    """
    if state.get("approved", False):
        log.info("[Quality Gate] APPROVED — pipeline complete")
        return "end"
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        log.warning("[Quality Gate] Max iterations (%d) reached — exiting without approval", MAX_ITERATIONS)
        return "end"
    log.info("[Quality Gate] REJECTED — sending back to architect for revision")
    return "requirement_architect"


# ─────────────────────────────────────────────
# 5. BUILD THE GRAPH
# ─────────────────────────────────────────────

def build_pipeline() -> StateGraph:
    graph = StateGraph(RequirementsState)

    graph.add_node("voc_generator", voc_generator_node)
    graph.add_node("voc_analyst", voc_analyst_node)
    graph.add_node("requirement_architect", requirement_architect_node)
    graph.add_node("quality_checker", quality_checker_node)

    graph.set_conditional_entry_point(
        should_generate_voc,
        {"voc_generator": "voc_generator", "voc_analyst": "voc_analyst"}
    )

    graph.add_edge("voc_generator", "voc_analyst")
    graph.add_edge("voc_analyst", "requirement_architect")
    graph.add_edge("requirement_architect", "quality_checker")

    graph.add_conditional_edges(
        "quality_checker",
        quality_gate,
        {"requirement_architect": "requirement_architect", "end": END}
    )

    return graph.compile()


# ─────────────────────────────────────────────
# 6. PUBLIC RUNNER (used by FastAPI and Streamlit)
# ─────────────────────────────────────────────

def run_pipeline(product_domain: str, voc_input: str = "", generate_voc: bool = False) -> dict:
    """
    Main entry point. Called by FastAPI.
    Returns the final state with all generated artifacts.
    """
    log.info("=" * 60)
    log.info("Pipeline start | domain='%s' | generate_voc=%s", product_domain, generate_voc)
    t_start = time.time()

    pipeline = build_pipeline()

    initial_state: RequirementsState = {
        "product_domain": product_domain,
        "generate_voc": generate_voc,
        "voc_input": voc_input,
        "stakeholder_needs": [],
        "epics": [],
        "features": [],
        "user_stories": [],
        "quality_feedback": "",
        "quality_score": None,
        "quality_issues": [],
        "approved": False,
        "iteration": 0
    }

    final_state = pipeline.invoke(initial_state)

    log.info(
        "Pipeline complete | approved=%s | iterations=%d | total=%.1fs",
        final_state["approved"], final_state["iteration"], time.time() - t_start
    )
    log.info("=" * 60)

    return {
        "voc_used": final_state["voc_input"],
        "stakeholder_needs": final_state["stakeholder_needs"],
        "epics": final_state["epics"],
        "features": final_state["features"],
        "user_stories": final_state["user_stories"],
        "quality_approved": final_state["approved"],
        "quality_score": final_state.get("quality_score"),
        "quality_issues": final_state.get("quality_issues", []),
        "iterations": final_state["iteration"]
    }


# ─────────────────────────────────────────────
# 7. LOCAL TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    result = run_pipeline(
        product_domain="hospital patient scheduling system",
        generate_voc=True
    )
    print(json.dumps(result, indent=2))