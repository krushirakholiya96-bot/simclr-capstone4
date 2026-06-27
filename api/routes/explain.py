from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from ai.generative import GenerativeAI

router = APIRouter()
groq = GenerativeAI()


class ExplainRequest(BaseModel):
    predicted_class: str
    confidence: float
    top5: List[dict]


class ExplainResponse(BaseModel):
    explanation: str
    confidence_analysis: str
    warning: Optional[str] = None


@router.post("/explain", response_model=ExplainResponse)
async def explain(request: ExplainRequest):
    # Generate explanation
    explanation = groq.generate_explanation(
        request.predicted_class,
        request.confidence,
        request.top5
    )

    # Analyze confidence
    confidence_analysis = groq.analyze_confidence(
        request.predicted_class,
        request.confidence
    )

    # Generate warning if low confidence
    warning = None
    if request.confidence < 60:
        warning = groq.generate_warning(
            request.predicted_class,
            request.confidence,
            request.top5
        )

    return ExplainResponse(
        explanation=explanation,
        confidence_analysis=confidence_analysis,
        warning=warning
    )