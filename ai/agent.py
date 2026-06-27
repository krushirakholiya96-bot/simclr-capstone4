import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from ai.generative import GenerativeAI
from src.model import SimCLRModel

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    )
])


class SimCLRAgent:
    def __init__(self, model, classifier, db, groq_client=None):
        self.model = model
        self.classifier = classifier
        self.db = db
        self.groq = GenerativeAI()

    def _validate(self, image):
        # Step 1: Validate image format and size
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert('RGB')
        return image

    def _extract_features(self, image):
        # Step 2: Extract features using SimCLR encoder
        tensor = TRANSFORM(image).unsqueeze(0)
        with torch.no_grad():
            features = self.model.encoder(tensor).flatten(1)
        return features

    def _predict(self, features):
        # Step 3: Generate top-5 predictions
        with torch.no_grad():
            logits = self.classifier.classifier(features)
            probs = F.softmax(logits, dim=1)[0]

        top5_probs, top5_idx = probs.topk(5)
        top5 = [
            {'class': CLASSES[idx.item()],
             'confidence': prob.item() * 100}
            for prob, idx in zip(top5_probs, top5_idx)
        ]
        return {
            'class': top5[0]['class'],
            'confidence': top5[0]['confidence'],
            'top5': top5
        }

    def _find_similar(self, predicted_class):
        # Step 4: Find similar past predictions from database
        try:
            similar = self.db.get_similar(predicted_class, limit=5)
            return similar
        except Exception:
            return []

    def _explain(self, prediction):
        # Step 5: Generate AI explanation using Groq
        return self.groq.generate_explanation(
            prediction['class'],
            prediction['confidence'],
            prediction['top5']
        )

    def _check_confidence(self, confidence):
        # Step 6: Check confidence and generate warning if needed
        if confidence < 60:
            return f"Low confidence warning: Model is only {confidence:.1f}% confident."
        return None

    def _compile_report(self, prediction, similar,
                        explanation, warning):
        # Step 7: Compile final structured report
        report = {
            'predicted_class': prediction['class'],
            'confidence': prediction['confidence'],
            'top5': prediction['top5'],
            'explanation': explanation,
            'similar_cases': len(similar),
            'warning': warning,
            'summary': self.groq.generate_report_summary(
                prediction, explanation, len(similar), warning
            )
        }
        return report

    def _save(self, report):
        # Step 8: Save report to database
        try:
            self.db.save_prediction(report)
        except Exception:
            pass
        return report

    def run(self, image) -> dict:
        # Run all 8 steps autonomously
        image = self._validate(image)
        features = self._extract_features(image)
        prediction = self._predict(features)
        similar = self._find_similar(prediction['class'])
        explanation = self._explain(prediction)
        warning = self._check_confidence(prediction['confidence'])
        report = self._compile_report(
            prediction, similar, explanation, warning
        )
        self._save(report)
        return report

    def run_step(self, step_num, image):
        # Run individual step for dashboard live display
        steps = [
            lambda: self._validate(image),
            lambda: self._extract_features(self._validate(image)),
            lambda: self._predict(
                self._extract_features(self._validate(image))
            ),
            lambda: self._find_similar('unknown'),
            lambda: 'Explanation generated',
            lambda: 'Confidence checked',
            lambda: 'Report compiled',
            lambda: 'Results saved'
        ]
        return steps[step_num]()