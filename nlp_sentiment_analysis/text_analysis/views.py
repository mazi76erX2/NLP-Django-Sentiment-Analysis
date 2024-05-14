from typing import Optional, Any, List, Dict

from rest_framework import status
from rest_framework.response import Response
from rest_framework.request import Request

from .analysis import analyse_sentiment_async
from .models import Analysis
from .serializers import AnalysisSerializer

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from adrf.views import APIView
import asyncio


class BulkAnalysisViewSet(APIView):
    """
    Async ViewSet for performing bulk sentiment analysis on a list of texts.

    This ViewSet provides the following actions:
    - post:
        Analyzes a list of texts asynchronously, creates analysis objects,
        and returns the analysis results.

    The post action expects a POST request with the following data:
    - texts: A list of texts (in string format) to be analyzed. (Optional[List[str]])

    The post action returns a response with the following data:
    - A list of analysis objects containing the sentiment analysis results.

    Example usage:
    POST /bulk-analysis/
    {
        "texts": ["I love this product!", "This movie is terrible."]
    }
    """

    permission_classes = [AllowAny]

    async def post(self, request: Request, *args: Any, **kwargs: Any) -> Response:
        texts: Optional[List[str]] = request.data.get("texts", [])

        if not texts:
            return Response(
                data={"error": 'Missing "texts" field in request data'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        tasks = [self.analyse_text(text) for text in texts]
        sentiment_results = await asyncio.gather(*tasks)

        analyses = [
            Analysis(
                text=text,
                sentiment=result['sentiment'],
                confidence_score=result['confidence_score']
            )
            for result, text in zip(sentiment_results, texts)
        ]
        Analysis.objects.bulk_create(analyses)

        serializer = AnalysisSerializer(analyses, many=True)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    async def analyse_text(self, text: str) -> Dict[str, float]:
        sentiment = await analyse_sentiment_async(text)
        return sentiment
