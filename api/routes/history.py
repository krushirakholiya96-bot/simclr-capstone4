from fastapi import APIRouter
from api.database.crud import Database

router = APIRouter()


@router.get("/history")
async def get_history():
    # Get last 50 predictions
    db = Database()
    predictions = db.get_history(limit=50)

    return [
        {
            'id': p.id,
            'image_name': p.image_name,
            'predicted_class': p.predicted_class,
            'confidence': p.confidence,
            'explanation': p.explanation,
            'warning': p.warning,
            'timestamp': str(p.timestamp)
        }
        for p in predictions
    ]


@router.delete("/history/{prediction_id}")
async def delete_prediction(prediction_id: int):
    # Delete prediction by ID
    db = Database()
    success = db.delete_prediction(prediction_id)

    if success:
        return {'success': True, 'message': 'Prediction deleted'}
    return {'success': False, 'message': 'Prediction not found'}