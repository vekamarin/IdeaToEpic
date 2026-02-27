"""
IdeaToEpic
Multi-agent LangGraph system that transforms VOC input into structured product backlogs.
"""

import os
import json
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
load_dotenv()

# ─────────────────────────────────────────────
# 1. SHARED STATE
# ─────────────────────────────────────────────

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
    approved: bool
    iteration: int                # safety counter to avoid infinite loops


# ─────────────────────────────────────────────
# 2. LLM SETUP
# ─────────────────────────────────────────────

def get_llm():
    """Initialize Gemini. Set GOOGLE_API_KEY in your .env file."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",   # free tier model
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3
    )


# ─────────────────────────────────────────────
# 3. AGENT NODES
# ─────────────────────────────────────────────

def voc_generator_node(state: RequirementsState) -> RequirementsState:
    """
    Generates a realistic VOC if the user didn't provide one.
    Only runs when generate_voc = True.
    """
    llm = get_llm()

    prompt = f"""You are a product discovery specialist. Generate a realistic 
Voice of Customer (VOC) input for a {state['product_domain']} product.

Include:
- 2-3 different user personas with distinct needs and roles
- Real pain points (specific, not generic)
- Desired outcomes, not just features
- Some conflicting needs between personas (this is realistic!)
- Written in natural conversational language, as if from stakeholder interviews

Do NOT write requirements or user stories. Write raw customer voice only.
Output plain text, no headers or formatting."""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {**state, "voc_input": response.content}


def voc_analyst_node(state: RequirementsState) -> RequirementsState:
    """
    Extracts structured stakeholder needs from raw VOC text.
    """
    llm = get_llm()

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
        needs = json.loads(response.content)
    except json.JSONDecodeError:
        # Fallback: wrap raw text if JSON parsing fails
        needs = [{"raw": response.content}]

    return {**state, "stakeholder_needs": needs}


def requirement_architect_node(state: RequirementsState) -> RequirementsState:
    """
    Builds the Epic → Feature → User Story hierarchy with full traceability.
    This is the core of the pipeline.
    """
    llm = get_llm()

    needs_text = json.dumps(state["stakeholder_needs"], indent=2)

    prompt = f"""You are a senior systems engineer specializing in requirements architecture.

Given these stakeholder needs, generate a fully traceable product backlog hierarchy.

STAKEHOLDER NEEDS:
{needs_text}

Quality feedback to address (if any): {state.get('quality_feedback', 'None')}

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
- Every ID must create a clear traceability chain
- Return ONLY valid JSON, no markdown, no explanation"""

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        hierarchy = json.loads(response.content)
        epics = hierarchy.get("epics", [])
    except json.JSONDecodeError:
        epics = [{"raw": response.content}]

    # Flatten features and stories for easy access
    features = []
    user_stories = []
    for epic in epics:
        for feature in epic.get("features", []):
            features.append(feature)
            for story in feature.get("user_stories", []):
                user_stories.append(story)

    return {**state, "epics": epics, "features": features, "user_stories": user_stories}


def quality_checker_node(state: RequirementsState) -> RequirementsState:
    """
    Reviews the full backlog for quality, gaps, and traceability.
    Returns APPROVED or REJECTED with specific feedback.
    """
    llm = get_llm()

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
  "score": 1-10,
  "issues": ["issue 1", "issue 2"],
  "feedback": "Specific instructions for improvement if REJECTED"
}}

Return ONLY valid JSON."""

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        result = json.loads(response.content)
        approved = result.get("status") == "APPROVED"
        feedback = result.get("feedback", "")
    except json.JSONDecodeError:
        approved = True  # if parsing fails, pass through
        feedback = ""

    return {
        **state,
        "approved": approved,
        "quality_feedback": feedback,
        "iteration": state.get("iteration", 0) + 1
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
    """Loop back to architect if quality fails. Max 2 retries."""
    if state.get("approved", False) or state.get("iteration", 0) >= 2:
        return "end"
    return "requirement_architect"


# ─────────────────────────────────────────────
# 5. BUILD THE GRAPH
# ─────────────────────────────────────────────

def build_pipeline() -> StateGraph:
    graph = StateGraph(RequirementsState)

    # Add nodes
    graph.add_node("voc_generator", voc_generator_node)
    graph.add_node("voc_analyst", voc_analyst_node)
    graph.add_node("requirement_architect", requirement_architect_node)
    graph.add_node("quality_checker", quality_checker_node)

    # Entry point with conditional routing
    graph.set_conditional_entry_point(
        should_generate_voc,
        {
            "voc_generator": "voc_generator",
            "voc_analyst": "voc_analyst"
        }
    )

    # Linear flow
    graph.add_edge("voc_generator", "voc_analyst")
    graph.add_edge("voc_analyst", "requirement_architect")
    graph.add_edge("requirement_architect", "quality_checker")

    # Quality gate — loop or end
    graph.add_conditional_edges(
        "quality_checker",
        quality_gate,
        {
            "requirement_architect": "requirement_architect",
            "end": END
        }
    )

    return graph.compile()


# ─────────────────────────────────────────────
# 6. PUBLIC RUNNER FUNCTION (used by FastAPI)
# ─────────────────────────────────────────────

def run_pipeline(
    product_domain: str,
    voc_input: str = "",
    generate_voc: bool = False
) -> dict:
    """
    Main entry point. Called by FastAPI.
    Returns the final state with all generated artifacts.
    """
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
        "approved": False,
        "iteration": 0
    }

    final_state = pipeline.invoke(initial_state)

    return {
        "voc_used": final_state["voc_input"],
        "stakeholder_needs": final_state["stakeholder_needs"],
        "epics": final_state["epics"],
        "features": final_state["features"],
        "user_stories": final_state["user_stories"],
        "quality_approved": final_state["approved"],
        "iterations": final_state["iteration"]
    }


# ─────────────────────────────────────────────
# 7. LOCAL TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Test with auto-generated VOC
    result = run_pipeline(
        product_domain="hospital patient scheduling system",
        generate_voc=True
    )
    print(json.dumps(result, indent=2))
