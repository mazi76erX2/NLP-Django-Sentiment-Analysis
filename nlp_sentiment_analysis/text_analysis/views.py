from typing import Optional, Any

from rest_framework import viewsets
from rest_framework import status
from rest_framework.response import Response
from rest_framework.request import Request

from .analysis import analyse_sentiment
from .models import Analysis
from .serializers import AnalysisSerializer


class BulkAnalysisViewSet(viewsets.ModelViewSet):
    """
    ViewSet for performing bulk sentiment analysis on a list of texts.

    This ViewSet provides the following actions:
    - create:
        Analyzes a list of texts, creates analysis objects,
        and returns the analysis results.

    The create action expects a POST request with the following data:
    - texts: A list of texts (in string format) to be analyzed. (Optional[List[str]])

    The create action returns a response with the following data:
    - A list of analysis objects containing the sentiment analysis results.

    Example usage:
    POST /bulk-analysis/
    {
        "texts": ["I love this product!", "This movie is terrible."]
    }
    """

    queryset: Analysis = Analysis.objects.all()
    serializer_class: AnalysisSerializer = AnalysisSerializer

    def create(self, request: Request, *args: Any, **kwargs: Any) -> Response:
        # Extract list of texts from request data
        texts: Optional[list[str]] = request.data.get("texts", [])

        if not texts:
            return Response(
                {"error": 'Missing "texts" field in request data'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        sentiment_results: list[dict[str, float]] = []
        for text in texts:
            sentiment = analyse_sentiment(text)
            sentiment_results.append(sentiment)

        # Create and save analysis objects
        analyses: list[Analysis] = []
        for result, text in zip(sentiment_results, texts):
            analysis: Analysis = Analysis(
                text=text,
                sentiment=result.sentiment,
                confidence_score=result.confidence_score,
            )
            analyses.append(analysis)
        Analysis.objects.bulk_create(analyses)

        # Serialize and return response
        serializer: AnalysisSerializer = AnalysisSerializer(analyses, many=True)
        return Response(serializer.data, status=status.HTTP_201_CREATED)
