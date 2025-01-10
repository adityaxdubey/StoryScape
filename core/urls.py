from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import StoryViewSet, StoryCreateView
from . import views # Ensure this import works without errors
from django.http import HttpResponseNotFound
router = DefaultRouter()
router.register('story', StoryViewSet, basename='story')

urlpatterns = [
    path('', views.index, name='index'),
    path('story/', StoryViewSet.as_view({'post': 'generate'})),
    path('story/generate/', StoryViewSet.as_view({'post': 'generate_story'}), name='generate_story'),
    path('story/<int:pk>/', views.story_detail, name='story_detail'),
    path('story/new/',  StoryCreateView.as_view(), name='story_create'),
    path('story/<int:pk>/edit/', views.story_update, name='story_update'),
    path('story_list/', views.story_list, name='story_list'),
    path('story/<int:pk>/delete/', views.story_delete, name='story_delete')
    
    
]
