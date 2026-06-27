from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class PredictionRequest(BaseModel):
    image_name: Optional[str] = 'unknown'


class Top5Item(BaseModel):
    class_name: str
    confidence: float


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    top5: List[dict]
    explanation: Optional[str] = None


class AgentReportResponse(BaseModel):
    predicted_class: str
    confidence: float
    top5: List[dict]
    explanation: str
    similar_cases: int
    warning: Optional[str] = None
    summary: str


class HistoryItem(BaseModel):
    id: int
    image_name: str
    predicted_class: str
    confidence: float
    explanation: str
    timestamp: datetime

    class Config:
        from_attributes = True


class DeleteResponse(BaseModel):
    success: bool
    message: str