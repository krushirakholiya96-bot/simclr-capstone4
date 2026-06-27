from fastapi import APIRouter, UploadFile, File
from PIL import Image
import io
from ai.agent import SimCLRAgent
from api.database.crud import Database

router = APIRouter()


@router.post("/agent/run")
async def run_agent(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')

    # Get model from app state
    from api.main import encoder, classifier
    db = Database()

    # Initialize agent
    agent = SimCLRAgent(
        model=encoder,
        classifier=classifier,
        db=db
    )

    # Run 8-step autonomous pipeline
    report = agent.run(image)

    return {
        'predicted_class': report['predicted_class'],
        'confidence': report['confidence'],
        'top5': report['top5'],
        'explanation': report['explanation'],
        'similar_cases': report['similar_cases'],
        'warning': report['warning'],
        'summary': report['summary']
    }