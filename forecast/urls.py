from django.urls import path
from .views import PredictionView
from .views import get_prediction

urlpatterns = [
    path('predictions/', PredictionView.as_view(), name='predictions'),
    # Include get_prediction directly in the urlpatterns
    path('get_prediction/', get_prediction,
         name='get_prediction'),
]
