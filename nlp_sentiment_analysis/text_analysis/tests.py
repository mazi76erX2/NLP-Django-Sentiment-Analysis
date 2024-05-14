import unittest
import asyncio
from unittest.mock import patch

from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf

from .analysis import analyse_sentiment_async  # Replace with actual import path
from typing import Dict, Union

from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
import pytest


MODEL: str = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(MODEL)
model: TFAutoModelForSequenceClassification = (
    TFAutoModelForSequenceClassification.from_pretrained(MODEL)
)
sentiment_labels: list[str] = ["negative", "neutral", "positive"]

@pytest.mark.asyncio
async def test_successful_analysis(self, mocker):
    """
    Tests successful sentiment analysis with mocked model output.
    """
    mocker.patch(
        TFAutoModelForSequenceClassification, "__call__", return_value={"logits": tf.constant([[0.1, 0.8, 0.1]]})
    )  # Simulate model output
    text = "This is a positive sentiment."
    result = await analyse_sentiment_async(text)

    assert result["sentiment"] == "positive"
    assert abs(result["confidence_score"] - 0.8) < 0.01
    mocker.patch.assert_called_once()  # Ensure model is called


@pytest.mark.asyncio
async def test_preprocessing_error(self):
    """
    Tests handling of a preprocessing error raised by the tokenizer.
    """
    text = "This text will cause a preprocessing error."
    with pytest.raises(ValueError):
        await analyse_sentiment_async(text)


pytest.mark.asyncio
async def test_out_of_range_error(self, mocker):
    """
    Tests handling of a TensorFlow Out-of-Range error during sentiment analysis.

    Mocking the OutOfRangeError allows us to test how the function behaves
    when encountering this specific error. The test injects the error and asserts
    that the returned dictionary contains the expected error message ("TensorFlow error").
    """
    mocker.patch(
        "tensorflow.errors.OutOfRangeError",
        side_effect=tf.errors.OutOfRangeError(
            node_def="SomeNode", op="SomeOp", message="Out-of-range error"
        ),
    )
    text = "This text will cause an Out-of-Range error."
    result = await analyse_sentiment_async(text)

    assert result["error"] == "TensorFlow error"


@pytest.mark.asyncio
async def test_invalid_argument_error(self, mocker):
    """
    Tests handling of a TensorFlow InvalidArgument error during sentiment analysis.

    Similar to the OutOfRangeError test, this test mocks the InvalidArgumentError
    and asserts that the function returns the appropriate error message.
    """
    mocker.patch(
        "tensorflow.errors.InvalidArgumentError",
        side_effect=tf.errors.InvalidArgumentError(
            node_def="SomeNode", op="SomeOp", message="Invalid argument"
        ),
    )
    text = "This text will cause an InvalidArgument error."
    result = await analyse_sentiment_async(text)

    assert result["error"] == "TensorFlow error"
