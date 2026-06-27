import os
from groq import Groq
from dotenv import load_dotenv
from ai.prompts import (
    get_explanation_prompt,
    get_confidence_prompt,
    get_warning_prompt,
    get_report_prompt
)

load_dotenv()


class GenerativeAI:
    def __init__(self):
        self.client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        self.model = 'llama3-8b-8192'

    def _call_api(self, prompt, max_tokens=200):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

    def generate_explanation(self, predicted_class, confidence, top5):
        # Generate explanation for prediction
        prompt = get_explanation_prompt(predicted_class, confidence, top5)
        return self._call_api(prompt)

    def analyze_confidence(self, predicted_class, confidence):
        # Analyze confidence level
        prompt = get_confidence_prompt(predicted_class, confidence)
        return self._call_api(prompt, max_tokens=150)

    def generate_warning(self, predicted_class, confidence, top5):
        # Generate warning for low confidence predictions
        prompt = get_warning_prompt(predicted_class, confidence, top5)
        return self._call_api(prompt, max_tokens=150)

    def generate_report_summary(self, prediction, explanation,
                                similar_count, warning):
        # Generate final report summary
        prompt = get_report_prompt(prediction, explanation,
                                   similar_count, warning)
        return self._call_api(prompt, max_tokens=200)