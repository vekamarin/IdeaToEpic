"""
FastAPI wrapper for the IdeaToEpic Requirements Generation Pipeline.
Run locally with: uvicorn api:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import uvicorn

from idea2epic import run_pipeline, get_llm, build_voc_prompt
from langchain_core.messages import HumanMessage

# ─────────────────────────────────────────────
# 1. APP SETUP
# ─────────────────────────────────────────────

app = FastAPI(
    title="IdeaToEpic — Requirements Generator API",
    description="Transforms VOC input into structured product backlogs using AI agents.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# 2. REQUEST & RESPONSE MODELS
# ─────────────────────────────────────────────

class GenerateRequest(BaseModel):
    product_domain: str = Field(
        ...,
        description="Short description of the product type",
        example="hospital patient scheduling system"
    )
    voc_input: Optional[str] = Field(
        default="",
        description="Raw VOC text. Leave empty if generate_voc is True.",
        example="Our nurses struggle to see real-time schedule updates..."
    )
    generate_voc: bool = Field(
        default=False,
        description="Set to True to auto-generate VOC from product_domain"
    )


class GenerateResponse(BaseModel):
    voc_used: str
    stakeholder_needs: list
    epics: list
    features: list
    user_stories: list
    quality_approved: bool
    quality_score: Optional[int]
    quality_issues: List[str]
    iterations: int


class VocOnlyRequest(BaseModel):
    product_domain: str = Field(
        ...,
        example="hospital patient scheduling system"
    )


# ─────────────────────────────────────────────
# 3. ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/")
def health_check():
    """Health check — useful for Railway/Render/fly.io deployment."""
    return {"status": "ok", "message": "IdeaToEpic API is running", "version": "2.0.0"}


@app.post("/generate", response_model=GenerateResponse)
def generate_requirements(request: GenerateRequest):
    """
    Main endpoint. Runs the full LangGraph multi-agent pipeline.
    Returns a structured backlog: epics → features → user stories.
    """
    if not request.generate_voc and not (request.voc_input or "").strip():
        raise HTTPException(
            status_code=400,
            detail="Provide either voc_input text or set generate_voc to True."
        )

    try:
        result = run_pipeline(
            product_domain=request.product_domain,
            voc_input=request.voc_input or "",
            generate_voc=request.generate_voc
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@app.post("/generate-voc-only")
def generate_voc_only(request: VocOnlyRequest):
    """
    Utility endpoint — generates only the VOC text for preview
    before committing to the full pipeline. Uses the same prompt
    as the internal voc_generator_node (single source of truth).
    """
    try:
        prompt = build_voc_prompt(request.product_domain)
        response = get_llm().invoke([HumanMessage(content=prompt)])
        return {"voc_text": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# 4. LOCAL DEV SERVER
# ─────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)