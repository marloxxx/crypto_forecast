from django.db import models


class Prediction(models.Model):
    predictions = models.JSONField(blank=True, null=True)
