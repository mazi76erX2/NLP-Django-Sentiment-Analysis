from transformers import AutoTokenizer, AutoModelForSequenceClassification

DictSentiment = dict[int, dict[str, float]]


def analyze_sentiment(text: str) -> dict[int, dict[str, float]]:
    """
    Analyzes text sentiment using a pre-trained RoBERTa model.

    Args:
        text: The text input for sentiment analysis (str).

    Returns:
        A dictionary containing sentiment scores for the provided text (dict[int, dict[str, float]]).
        The dictionary key is the predicted sentiment class (int),
        and the value is another dictionary with "polarity" and "subjectivity" scores (float).
    """
    model_name: str = "roberta-base"  # Pre-trained model name (str)

    # Load tokenizer and model
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name)
    model: AutoModelForSequenceClassification = (
        AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    )  # 2 for binary sentiment

    # Tokenize and pad the text
    encoded_text = tokenizer(
        text, return_tensors="pt", padding="max_length", truncation=True
    )  # Encoded text (likely a tensor)

    # Perform prediction with torch.no_grad() for efficiency
    with torch.no_grad():
        outputs = model(**encoded_text)
        logits = outputs.logits  # Logits tensor (likely a float tensor)
        predictions = torch.argmax(logits, dim=-1).item()  # Predicted class (int)

    # Map prediction to sentiment scores (adjust based on model output)
    sentiment: DictSentiment = {
        0: {"polarity": -1.0, "subjectivity": 0.0},  # Negative
        1: {"polarity": 1.0, "subjectivity": 0.0},  # Positive
    }

    return sentiment[predictions]
