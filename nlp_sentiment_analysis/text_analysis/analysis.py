import string

from transformers import TFBertForSequenceClassification
import tensorflow as tf
from tensorflow_text import WordpieceTokenizer


DictSentiment = dict[int, dict[str, float]]
TupleProcessedTokens = tuple[list[str], list[int]]

MODEL_NAME = "bert/base-uncased-sentiment"
MAX_LEN = 512


def load_model(model_name=MODEL_NAME) -> TFBertForSequenceClassification:
    """
    Loads a pre-trained BERT model for sentiment classification.
    """
    return TFBertForSequenceClassification.from_pretrained(model_name, num_labels=2)


def create_tokenizer(model: TFBertForSequenceClassification) -> WordpieceTokenizer:
    """
    Creates a WordpieceTokenizer based on the model's vocabulary.
    """
    return WordpieceTokenizer(vocabulary_list=model.config.vocab_file.content)


def preprocess_text(
    text: str, model: TFBertForSequenceClassification
) -> TupleProcessedTokens:
    """
    Preprocesses text for sentiment analysis using a BERT model.
    """
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    tokenizer = create_tokenizer(model)
    tokens = tokenizer.tokenize(text)

    padding = "max_length" if len(tokens) < MAX_LEN else "post_truncation"
    padded_tokens = tf.keras.preprocessing.sequence.pad_sequences(
        [tokens], maxlen=MAX_LEN, padding=padding, truncating="post"
    ).tolist()[0]
    return padded_tokens


def predict_sentiment(
    model: TFBertForSequenceClassification, text: str
) -> DictSentiment:
    """
    Predicts sentiment for a given text using the loaded BERT model.
    """
    encoded_text = preprocess_text(text, model)
    outputs = model(encoded_text)
    logits = outputs.logits
    predictions = tf.math.argmax(logits, axis=-1).numpy()[0]

    # Sentiment mapping (adjust based on model output)
    sentiment = {
        0: {"polarity": -1.0, "subjectivity": 0.0},  # Negative
        1: {"polarity": 1.0, "subjectivity": 0.0},  # Positive
    }

    predicted_class = predictions
    sentiment_scores = sentiment[predicted_class]
    return {"text_sentiment": sentiment_scores}


def analyze_sentiment(text: str) -> DictSentiment:
    """
    Analyzes text sentiment using a pre-trained BERT model from TensorFlow Hub.

    Args:
        text: The text input for sentiment analysis (str).

    Returns:
        A dictionary containing sentiment scores for the provided text DictSentiment.
        The dictionary key is the predicted sentiment class (int),
        and the value is another dictionary with "polarity" and "subjectivity" scores (float).
    """
    model = load_model()
    sentiment_scores = predict_sentiment(model, text)
    return sentiment_scores
