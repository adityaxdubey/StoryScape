from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import CharacterViewSet, DialogueViewSet

router = DefaultRouter()
router.register(r'characters', CharacterViewSet)
router.register(r'dialogues', DialogueViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
