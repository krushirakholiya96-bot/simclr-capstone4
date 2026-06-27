import json
from datetime import datetime
from api.database.db import SessionLocal, Prediction


class Database:
    def __init__(self):
        self.db = SessionLocal()

    def save_prediction(self, report: dict, image_name: str = 'unknown'):
        # Save prediction to database
        prediction = Prediction(
            image_name=image_name,
            predicted_class=report['predicted_class'],
            confidence=report['confidence'],
            top5=json.dumps(report['top5']),
            explanation=report.get('explanation', ''),
            warning=report.get('warning', None),
            report=report.get('summary', ''),
            timestamp=datetime.utcnow()
        )
        self.db.add(prediction)
        self.db.commit()
        self.db.refresh(prediction)
        return prediction

    def get_history(self, limit: int = 50):
        # Get last N predictions
        predictions = self.db.query(Prediction)\
            .order_by(Prediction.timestamp.desc())\
            .limit(limit)\
            .all()
        return predictions

    def get_similar(self, predicted_class: str, limit: int = 5):
        # Get similar past predictions by class
        predictions = self.db.query(Prediction)\
            .filter(Prediction.predicted_class == predicted_class)\
            .order_by(Prediction.timestamp.desc())\
            .limit(limit)\
            .all()
        return predictions

    def delete_prediction(self, prediction_id: int):
        # Delete a prediction by ID
        prediction = self.db.query(Prediction)\
            .filter(Prediction.id == prediction_id)\
            .first()
        if prediction:
            self.db.delete(prediction)
            self.db.commit()
            return True
        return False

    def get_prediction_by_id(self, prediction_id: int):
        # Get single prediction by ID
        return self.db.query(Prediction)\
            .filter(Prediction.id == prediction_id)\
            .first()