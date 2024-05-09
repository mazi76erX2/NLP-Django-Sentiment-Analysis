from transformers import TFBertForSequenceClassification, TFBertTokenizer
import tensorflow as tf

DictSentiment = dict[int, dict[str, float]]

def analyze_sentiment(text: str) -> DictSentiment:
  """
  Analyzes text sentiment using a pre-trained BERT model from TensorFlow Hub.

  Args:
      text: The text input for sentiment analysis (str).

  Returns:
      A dictionary containing sentiment scores for the provided text (dict[int, dict[str, float]]).
      The dictionary key is the predicted sentiment class (int),
      and the value is another dictionary with "polarity" and "subjectivity" scores (float).
  """
  model_name: str = "bert/base-uncased-sentiment"  # Pre-trained model name (str)

  # Load tokenizer and model
  tokenizer: TFBertTokenizerFast = TFBertTokenizerFast.from_pretrained(model_name)
  model: TFBertForSequenceClassification = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

  # Tokenize and pad the text
  encoded_text: dict[str, tf.Tensor] = tokenizer(
      text, return_tensors="tf", padding="max_length", truncation=True
  )

  # Perform prediction
  with tf.device("/CPU:0"):  # Use CPU for efficiency (adjust as needed)
    outputs: tf.Tensor = model(encoded_text)
    logits: tf.Tensor = outputs.logits
    predictions: int = tf.math.argmax(logits, axis=-1).numpy()[0]  # Get single prediction

  # Map prediction to sentiment scores (adjust based on model output)
  sentiment: DictSentiment = {
      0: {"polarity": -1.0, "subjectivity": 0.0},  # Negative
      1: {"polarity": 1.0, "subjectivity": 0.0},  # Positive
  }

  return sentiment[predictions]
