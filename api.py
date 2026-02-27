"""
FastAPI wrapper for the Requirements Generation Pipeline.
Run locally with: uvicorn api:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn

from idea2epic import run_pipeline

# ─────────────────────────────────────────────
# 1. APP SETUP
# ─────────────────────────────────────────────

app = FastAPI(
    title="Requirements Generator API",
    description="Transforms VOC input into structured product backlogs using AI agents.",
    version="1.0.0"
)

# Allow Lovable / any frontend to call this API
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
        description="Raw VOC text written by the user. Leave empty if generate_voc is True.",
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
    iterations: int


# ─────────────────────────────────────────────
# 3. ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/")
def health_check():
    """Quick health check — useful for Railway/Render deployment."""
    return {"status": "ok", "message": "Requirements Generator API is running"}


@app.post("/generate", response_model=GenerateResponse)
def generate_requirements(request: GenerateRequest):
    """
    Main endpoint. Accepts VOC input or auto-generates one.
    Runs the full LangGraph multi-agent pipeline and returns
    a structured backlog with epics, features, and user stories.
    """

    # Validate: must have either voc_input or generate_voc=True
    if not request.generate_voc and not request.voc_input.strip():
        raise HTTPException(
            status_code=400,
            detail="Provide either voc_input text or set generate_voc to True."
        )

    try:
        result = run_pipeline(
            product_domain=request.product_domain,
            voc_input=request.voc_input,
            generate_voc=request.generate_voc
        )
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(e)}"
        )


@app.post("/generate-voc-only")
def generate_voc_only(product_domain: str):
    """
    Utility endpoint — generates only the VOC text for preview
    before running the full pipeline. Useful for the UI toggle flow.
    """
    from idea2epic import get_llm
    from langchain_core.messages import HumanMessage

    llm = get_llm()

    prompt = f"""Generate a realistic Voice of Customer (VOC) input for a {product_domain} product.
Include 2-3 personas, their pain points, and desired outcomes.
Write in natural conversational language. No headers or formatting."""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"voc_text": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# 4. LOCAL DEV SERVER
# ─────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
