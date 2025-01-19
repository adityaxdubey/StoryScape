from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import CharacterViewSet, DialogueViewSet, generate_character_image

router = DefaultRouter()
router.register(r'characters', CharacterViewSet)
router.register(r'dialogues', DialogueViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('characters/<int:character_id>/generate-image/', 
         generate_character_image, 
         name='generate_character_image'),
]
