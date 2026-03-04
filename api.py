"""
FastAPI wrapper for the IdeaToEpic Requirements Generation Pipeline.
Run locally with: uvicorn api:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import json
import asyncio
import traceback

from idea2epic import run_pipeline, llm, build_voc_prompt, build_streaming_pipeline
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
    iteration_history: Optional[List[dict]] = []


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


@app.post("/generate-voc-only")
def generate_voc_only(request: VocOnlyRequest):
    """
    Utility endpoint — generates only the VOC text for preview
    before committing to the full pipeline. Uses the same prompt
    as the internal voc_generator_node (single source of truth).
    """
    try:
        prompt = build_voc_prompt(request.product_domain)
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"voc_text": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-stream")
async def generate_requirements_stream(request: GenerateRequest):
    """
    Streaming endpoint. Emits Server-Sent Events (SSE) as the pipeline progresses.
    """
    if not request.generate_voc and not (request.voc_input or "").strip():
        raise HTTPException(
            status_code=400,
            detail="Provide either voc_input text or set generate_voc to True."
        )

    async def event_generator():
        try:
            pipeline = build_streaming_pipeline()
            
            initial_state = {
                "product_domain": request.product_domain,
                "generate_voc": request.generate_voc,
                "voc_input": request.voc_input or "",
                "stakeholder_needs": [],
                "epics": [],
                "features": [],
                "user_stories": [],
                "quality_feedback": "",
                "quality_score": None,
                "quality_issues": [],
                "approved": False,
                "iteration": 0,
                "iteration_history": []  # ✅ Added
            }
            
            seen_events = set()  # ✅ Changed from last_node
            last_state = None
            
            async for state_update in pipeline.astream(initial_state):
                for node_name, current_state in state_update.items():
                    # ✅ Track by (node, iteration) instead of just node
                    iteration = current_state.get("iteration", 0)
                    event_key = (node_name, iteration)
                    
                    if event_key in seen_events:
                        continue
                    
                    seen_events.add(event_key)
                    
                    status = {
                        "node": node_name,
                        "needs_count": len(current_state.get("stakeholder_needs", [])),
                        "epics_count": len(current_state.get("epics", [])),
                        "features_count": len(current_state.get("features", [])),
                        "stories_count": len(current_state.get("user_stories", [])),
                        "iteration": iteration,
                        "approved": current_state.get("approved", False),
                        "quality_score": current_state.get("quality_score")
                    }
                    
                    yield f"event: node_complete\n"
                    yield f"data: {json.dumps(status)}\n\n"
                    await asyncio.sleep(0.01)
                    
                    last_state = current_state
            
            # ✅ Fixed: only yield if we have state
            if last_state:
                final_result = {
                    "voc_used": last_state["voc_input"],
                    "stakeholder_needs": last_state["stakeholder_needs"],
                    "epics": last_state["epics"],
                    "features": last_state["features"],
                    "user_stories": last_state["user_stories"],
                    "quality_approved": last_state["approved"],
                    "quality_score": last_state.get("quality_score"),
                    "quality_issues": last_state.get("quality_issues", []),
                    "iterations": last_state["iteration"],
                    "iteration_history": last_state.get("iteration_history", [])
                }
                
                yield f"event: final_result\n"
                yield f"data: {json.dumps(final_result)}\n\n"
            else:
                yield f"event: error\n"
                yield f"data: {json.dumps({'error': 'Pipeline completed with no state'})}\n\n"
            
        except Exception as e:
            # ✅ Better error reporting
            error_details = {
                'error': str(e),
                'type': type(e).__name__,
                'traceback': traceback.format_exc()
            }
            print(f"Stream error:\n{error_details['traceback']}")
            
            yield f"event: error\n"
            yield f"data: {json.dumps(error_details)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )



# ─────────────────────────────────────────────
# 4. LOCAL DEV SERVER
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)