# All prompts for Generative AI and Agentic AI


def get_explanation_prompt(predicted_class, confidence, top5):
    return f"""
    A SimCLR self-supervised model (ResNet50 backbone) classified
    a CIFAR-10 image as: {predicted_class}
    Confidence: {confidence:.1f}%
    Top 5 predictions: {top5}

    Explain in 3-4 simple sentences what visual features
    the model likely detected to make this classification.
    Be specific about colors, shapes, textures, and patterns.
    """


def get_confidence_prompt(predicted_class, confidence):
    return f"""
    A SimCLR model predicted: {predicted_class}
    Confidence: {confidence:.1f}%

    Analyze this confidence level in 2 sentences.
    Is this a reliable prediction? What might cause uncertainty?
    """


def get_warning_prompt(predicted_class, confidence, top5):
    return f"""
    A SimCLR model made a low confidence prediction.
    Predicted class: {predicted_class}
    Confidence: {confidence:.1f}%
    Top 5 predictions: {top5}

    In 2 sentences, explain why this prediction might be unreliable
    and what the user should do.
    """


def get_report_prompt(prediction, explanation, similar_count, warning):
    return f"""
    Generate a brief 3-sentence summary report for this image analysis:
    Prediction: {prediction['class']} ({prediction['confidence']:.1f}% confidence)
    AI Explanation: {explanation}
    Similar past cases found: {similar_count}
    Warning: {warning if warning else 'None'}

    Make it professional and informative.
    """