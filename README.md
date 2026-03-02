# IdeaToEpic

**Transform raw customer voice into structured product backlogs using a multi-agent AI pipeline.**

IdeaToEpic is a LangGraph-powered system that takes Voice of Customer (VOC) input — either written by you or auto-generated — and produces a fully traceable product backlog with Epics, Features, and User Stories, complete with acceptance criteria and a quality gate.

---

## Why I Built This

Requirements engineering is where most product development goes wrong. Stakeholder needs get lost in translation, backlogs are created without traceability, and quality checks happen too late.

This project explores whether a multi-agent AI system can replicate and accelerate the work of a senior business analyst and systems architect working in tandem. It's also a hands-on vehicle for learning agent orchestration, LangGraph state management, and structured output generation with LLMs.

---

## Architecture

```
VOC Input (user-provided or auto-generated)
        │
        ▼
┌─────────────────────┐
│   VOC Generator     │  (optional) Generates realistic stakeholder interviews
│   Agent             │  with personas, pain points, and conflicting needs
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   VOC Analyst       │  Extracts structured stakeholder needs from raw text
│   Agent             │  → persona, pain point, desired outcome, priority
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Requirements      │  Builds Epic → Feature → User Story hierarchy
│   Architect Agent   │  with full ID traceability and acceptance criteria
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Quality Checker   │  Audits backlog for gaps, ambiguity, and traceability
│   Agent             │  → APPROVED or REJECTED with specific feedback
└─────────────────────┘
        │
    REJECTED? ──────────────────────► Back to Architect (max 2 retries for this demo)
        │
    APPROVED
        │
        ▼
   Final Backlog
```

Built with **LangGraph** for stateful agent orchestration and conditional routing. The quality feedback loop is the core architectural decision: the system doesn't just generate output, it reviews and iterates on it.

---

## Sample Output

Input: `hospital patient scheduling system` (VOC auto-generated)

```
✅ Quality Score: 9/10 — APPROVED
📦 4 Epics
🧩 12 Features  
📝 21 User Stories
🔁 Iterations: 1
```

Example user story generated:
```
US2.1.1
As a Head Nurse, I want to receive instant push notifications when a 
shift change affects my ward so that I can reassign coverage before 
patient care is impacted.

Acceptance Criteria:
- Given a schedule change is saved, When it affects an active ward,
  Then all assigned nurses receive a push notification within 30 seconds.
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Orchestration | LangGraph |
| LLM | Groq (Llama 3.3) (swapabble) |
| API | FastAPI |
| Frontend | Lovable (production UI) / Streamlit (dev interface) |
| Tracing | LangSmith |
| Env Management | python-dotenv |

---

## Project Structure

```
idea2epic/
├── idea2epic.py       # Core pipeline: agents, state, graph definition
├── api.py             # FastAPI wrapper with /generate and /generate-voc-only endpoints
├── app.py             # Stremlit app
├── .env               # API keys (not committed)
└── README.md
```

---

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Environment

Create a `.env` file:

```
GROQ_API_KEY=your_key_here
```

### Run the pipeline directly

```bash
python idea2epic.py
```

### Run the API server

```bash
uvicorn api:app --reload
```

API will be available at `http://localhost:8000`  
Interactive docs at `http://localhost:8000/docs`

### Run the Streamlit dev interface

```bash
streamlit run streamlit_app.py
```

A developer-friendly UI for testing the pipeline locally — useful for iterating on prompts and inspecting agent output without touching the API directly.

### API Usage

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "product_domain": "hospital patient scheduling system",
    "generate_voc": true
  }'
```

Or with your own VOC:

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "product_domain": "hospital patient scheduling system",
    "voc_input": "Our nurses can never see schedule changes in real time..."
  }'
```

---

## Key Design Decisions

**Why LangGraph?** LangGraph gives explicit control over state transitions and conditional routing.

**Why a quality gate?** Single-pass generation produces plausible but often shallow backlogs. The auditor agent catches untestable acceptance criteria, missing traceability, and uncovered stakeholder needs that the architect missed. This mirrors how real requirements processes work.

**Why structured JSON throughout?** Each agent outputs typed JSON, not prose. This enforces traceability between agents and makes the output directly usable by downstream tools (Jira, Confluence, etc.).

**LLM-agnostic design:** The `get_llm()` function is the single swap point. Gemini, GPT-4, Claude, or Groq can be substituted without touching agent logic.

---

## What's Next

- [ ] Deploy FastAPI backend to Railway/Render for live demo
- [ ] Jira export endpoint
- [ ] Multi-domain backlog merging

---

## Author

**Verônica Marin Kramer** — Senior Systems Engineer & Technical Delivery Lead transitioning into AI Product Management.  
[LinkedIn](https://linkedin.com/in/vekamarin) · [Email](mailto:veronica.marin.kramer@gmail.com)
