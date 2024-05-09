from django.urls import path, include
from rest_framework import permissions
from rest_framework.routers import DefaultRouter
from rest_framework_swagger import views as swagger_views

from .views import BulkAnalysisViewSet


router = DefaultRouter()
router.register(r"analyses", BulkAnalysisViewSet, basename="analyses")

urlpatterns = [
    path("", include(router.urls)),
]
