from typing import Optional, Any, List, Dict

import asyncio

from rest_framework import status
from rest_framework.response import Response
from rest_framework.request import Request
from rest_framework.permissions import AllowAny

from adrf.viewsets import ViewSet

from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi


from .analysis import analyse_sentiment_async
from .models import Analysis
from .serializers import AnalysisSerializer


class BulkAnalysisViewSet(ViewSet):
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

    queryset = Analysis.objects.all()
    serializer_class = AnalysisSerializer
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            required=["texts"],
            properties={
                "texts": openapi.Schema(
                    type=openapi.TYPE_ARRAY,
                    items=openapi.Schema(type=openapi.TYPE_STRING),
                )
            },
        )
    )
    async def create(self, request: Request, *args: Any, **kwargs: Any) -> Response:
        texts: Optional[List[str]] = request.data.get("texts", [])

        texts = ["I hate this product!", "This movie is great."]

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
        await Analysis.objects.abulk_create(analyses)

        serializer = AnalysisSerializer(analyses, many=True)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    async def analyse_text(self, text: str) -> Dict[str, float]:
        sentiment = await analyse_sentiment_async(text)
        return sentiment
