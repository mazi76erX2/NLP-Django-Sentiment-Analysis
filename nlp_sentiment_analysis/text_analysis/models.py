from django.db import models


class Analysis(models.Model):
    """
    This model stores the results of text sentiment analysis.

    Fields:
        text (str): The analyzed text.
        polarity (Optional[float]):
            The polarity score (-1.0 for negative, 1.0 for positive).
        subjectivity (Optional[float]):
            The subjectivity score (0.0 for objective, 1.0 for subjective).
        created_at (models.DateTimeField):
            The timestamp when the analysis was created. Automatically set on creation.
    """

    text = models.TextField(blank=False, null=False)
    polarity = models.FloatField(blank=True, null=True)
    subjectivity = models.FloatField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return f"{self.text[:20]}..." if self.text is not None else ""
