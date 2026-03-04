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
MAX_ITERATIONS = 5 # 1 initial attempt + 4 revision cycles if not stuck in score

# LLM setup - replace with your preferred model
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

# Utility function to clean JSON output from LLM
def clean_json(text: str) -> str:
    """Remove markdown code fences from LLM responses."""
    return text.replace("```json", "").replace("```", "").strip()

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
    iteration_history: list  # Track each iteration's score and issues

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
    
    # Show previous attempt if this is a revision
    if state.get("quality_feedback") and state.get("epics"):
        issues_list = "\n".join([f"{i+1}. {issue}" for i, issue in enumerate(state.get("quality_issues", []))])
        
        previous_attempt = f"""
════════════════════════════════════════════════════════════════
REVISION REQUIRED - Iteration {iteration} of {MAX_ITERATIONS}
Previous Score: {state.get('quality_score', 'N/A')}/10 (Target: 9+)
════════════════════════════════════════════════════════════════

YOUR PREVIOUS BACKLOG:
{json.dumps({"epics": state["epics"]}, indent=2)}

CRITICAL ISSUES FOUND (YOU MUST FIX THESE):
{issues_list}

DETAILED FIX INSTRUCTIONS:
{state['quality_feedback']}

════════════════════════════════════════════════════════════════
HOW TO REVISE (READ CAREFULLY):
════════════════════════════════════════════════════════════════

For COVERAGE GAP issues:
→ Read the exact persona + pain point mentioned
→ ADD a NEW user story with a SPECIFIC user action
→ DO NOT try to address emotional/cultural issues with software

EXAMPLE OF WHAT TO DO:
Quality checker says: "No story for scheduler to view daily appointment metrics"
You ADD:
{{
  "id": "US3.2.3",
  "feature_id": "F3.2",
  "story": "As a Central Scheduler, I want to view daily appointment booking metrics on my dashboard so that I can identify peak call times",
  "acceptance_criteria": [
    "Given I open my dashboard, When the page loads, Then I see total appointments booked, cancelled, and rescheduled for today",
    "Given I view the metrics, When I hover over the data, Then I see an hourly breakdown of booking activity"
  ]
}}

EXAMPLE OF WHAT NOT TO DO:
Quality checker says: "Patient feels like a burden"
You ADD: "As a Patient, I want to not feel like a burden..." ← THIS IS WRONG
Instead, skip emotional issues or translate them to functional actions:
"As a Patient, I want to book appointments online without calling..."

For CLARITY/TESTABILITY issues:
→ Find the EXACT story ID mentioned (e.g., "US1.2.3")
→ REWRITE only that story's text or acceptance criteria
→ Keep the same ID
→ Don't touch other stories

For PARTIAL ADDRESS issues:
→ The story exists but doesn't fully solve the problem
→ ENHANCE the existing story's acceptance criteria
→ Add additional acceptance criteria that cover the missing aspect

CRITICAL RULES:
1. Do NOT regenerate everything from scratch
2. Do NOT remove stories that weren't criticized
3. Do NOT change story IDs unless adding new stories
4. Keep all the GOOD work from your previous attempt
5. Make TARGETED fixes to the specific issues mentioned

Return the COMPLETE updated backlog as JSON.
"""
    else:
        previous_attempt = "This is your first attempt. Generate a complete, high-quality backlog."

    prompt = f"""You are a senior systems engineer specializing in requirements architecture.

Given these stakeholder needs, generate a fully traceable product backlog hierarchy.

STAKEHOLDER NEEDS:
{needs_text}

{previous_attempt}

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
                "Given [context], When [action], Then [outcome with measurable result]"
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
- Acceptance criteria MUST be measurable - include numbers, timeframes, or specific states
- Cover ALL stakeholder personas mentioned in the needs
- Each acceptance criterion must have "Given/When/Then" format with measurable outcome
- Return ONLY valid JSON, no markdown, no explanation

MEASURABILITY EXAMPLES:
❌ BAD: "System is fast"
✅ GOOD: "Page loads in under 2 seconds for 95% of requests"

❌ BAD: "User interface is intuitive"  
✅ GOOD: "New users complete first booking without help documentation in under 5 minutes"

❌ BAD: "Notifications work properly"
✅ GOOD: "Push notification appears on user's device within 30 seconds of schedule change"
"""

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
    On revisions, FIRST checks if previous issues were fixed.
    """
    log.info("[Quality Checker] Running quality audit (check #%d)", state.get("iteration", 0) + 1)
    t0 = time.time()

    backlog_summary = {
        "epics_count": len(state["epics"]),
        "features_count": len(state["features"]),
        "stories_count": len(state["user_stories"]),
        "epics": state["epics"]
    }
    
    # Check if this is a revision
    previous_issues = state.get("quality_issues", [])
    revision_context = ""
    if previous_issues:
        revision_context = f"""
════════════════════════════════════════════════════════════════
THIS IS A REVISION - PREVIOUS ISSUES WERE:
════════════════════════════════════════════════════════════════
{json.dumps(previous_issues, indent=2)}

CRITICAL: Before finding NEW issues, you MUST verify whether these previous issues were FIXED.
If an issue was NOT fixed, include it again in your response with "STILL UNFIXED:" prefix.
Only look for new issues AFTER confirming old ones are resolved.
════════════════════════════════════════════════════════════════
"""

    prompt = f"""You are a requirements quality auditor with 15+ years of experience.

Review this product backlog against the original stakeholder needs.

ORIGINAL STAKEHOLDER NEEDS:
{json.dumps(state['stakeholder_needs'], indent=2)}

GENERATED BACKLOG:
{json.dumps(backlog_summary, indent=2)}

{revision_context}

Audit criteria:
1. Traceability: Every story ID must reference a valid feature ID, every feature ID must reference an epic ID
2. Testability: Acceptance criteria must use Given/When/Then with measurable outcomes (numbers, timeframes, states)
3. Coverage: Every stakeholder persona and FUNCTIONAL pain point must be addressed by at least one user story
   - Focus on WHAT the user can DO differently, not emotional outcomes
   - Example: "Can't see schedule" needs a story for viewing schedules (good)
   - Example: "Feels like a burden" is too abstract - skip these (bad)
4. Clarity: User stories must follow "As a [specific role], I want [specific action] so that [specific benefit]"
5. Completeness: No placeholder text like "TBD", "etc", or vague outcomes

CRITICAL: Focus ONLY on functional gaps that can be solved with features.
Do NOT flag emotional/cultural/organizational issues that cannot be addressed by software alone.

Examples of GOOD coverage issues:
✅ "No story allows patient to view family appointments"
✅ "No story validates referral form completeness"

Examples of BAD coverage issues (skip these):
❌ "Patient feels like a burden" (emotional state, not a functional gap)
❌ "Scheduler overwhelmed by workload" (organizational problem, not software gap)

Scoring guide:
- 9-10: Production-ready, all personas covered, all acceptance criteria measurable
- 7-8: Good structure, minor gaps (1-2 personas with partial coverage OR 1-2 vague criteria)
- 5-6: Significant gaps (missing personas OR multiple untestable stories)
- 3-4: Major structural issues
- 1-2: Unusable

CRITICAL INSTRUCTIONS:
- Be SPECIFIC: Don't say "some stories are vague" - say "US1.2.1: acceptance criteria 'easy to use' is not measurable"
- For coverage gaps, cite the exact persona and pain point from STAKEHOLDER NEEDS
- Focus on the TOP 3-5 most critical issues only
- If this is a revision, FIRST check if previous issues were fixed before finding new ones

Return JSON:
{{
  "status": "APPROVED" or "REJECTED",
  "score": <integer 1-10>,
  "issues": ["Specific issue with story ID and fix needed", ...],
  "feedback": "Concise, actionable fixes with exact IDs and what to change"
}}

Return ONLY valid JSON, no markdown."""

    response = llm.invoke([HumanMessage(content=prompt)])
    new_iteration = state.get("iteration", 0) + 1

    try:
        result = json.loads(clean_json(response.content))
        approved = result.get("status") == "APPROVED"
        feedback = result.get("feedback", "")
        score = result.get("score")
        all_issues = result.get("issues", [])
        
        # PRIORITIZE: Coverage gaps, then testability, then clarity
        coverage_issues = [i for i in all_issues if "COVERAGE" in i.upper() or "GAP" in i.upper()]
        testability_issues = [i for i in all_issues if "TESTABILITY" in i.upper() or "TESTABLE" in i.upper() or "MEASURABLE" in i.upper()]
        other_issues = [i for i in all_issues if i not in coverage_issues and i not in testability_issues]
        
        # Take top 3 most critical
        prioritized_issues = (coverage_issues + testability_issues + other_issues)[:3]
        
        log.info(
            "[Quality Checker] Status: %s | Score: %s/10 | Total Issues: %d | Prioritized: %d | Iteration: %d",
            result.get("status"), score, len(all_issues), len(prioritized_issues), new_iteration
        )
        log.info("[Quality Checker] Issues: %s", prioritized_issues)
        
        issues = prioritized_issues
        
    except json.JSONDecodeError:
        log.warning("[Quality Checker] JSON parse failed — defaulting to REJECTED for safety")
        approved = False
        feedback = "Quality check response could not be parsed. Please regenerate with valid JSON."
        score = None
        issues = ["Quality auditor response was malformed"]

    log.info("[Quality Checker] Done in %.1fs", time.time() - t0)

    history_entry = {
        "iteration": new_iteration,
        "score": score,
        "status": result.get("status") if 'result' in locals() else "ERROR",
        "issues": issues,
        "timestamp": time.time()
    }
    current_history = state.get("iteration_history", [])
    current_history.append(history_entry)

    return {
        **state,
        "approved": approved,
        "quality_feedback": feedback,
        "quality_score": score,
        "quality_issues": issues,
        "iteration": new_iteration,
        "iteration_history": current_history
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
    Force exit if score is stagnating (same score for 3+ iterations).
    """
    if state.get("approved", False):
        log.info("[Quality Gate] APPROVED — pipeline complete")
        return "end"
    
    iteration = state.get("iteration", 0)
    
    # Check for stagnation (same score 2+ times)
    history = state.get("iteration_history", [])
    if len(history) > 2:
        last_two_scores = [h.get("score") for h in history[-2:]]
        if last_two_scores[0] == last_two_scores[1] and last_two_scores[0] is not None:
            log.warning(
                "[Quality Gate] Score stagnated at %d/10 for 2 iterations — forcing exit",
                last_two_scores[0]
            )
            return "end"
    
    if iteration >= MAX_ITERATIONS:
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


# Alias for streaming - same graph, just used with astream() instead of invoke()
build_streaming_pipeline = build_pipeline


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
        "iteration": 0,
        "iteration_history": []
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
        "iterations": final_state["iteration"],
        "iteration_history": final_state.get("iteration_history", [])
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