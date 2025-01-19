from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from .models import Character, CharacterDialogue, Story
from .serializers import CharacterSerializer, CharacterDialogueSerializer
from .services.character_generator import CharacterGenerator
from .services.dialogue_generator import DialogueGenerator
from django.shortcuts import render
from django.contrib import messages
import logging
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

logger = logging.getLogger(__name__)

class CharacterViewSet(viewsets.ModelViewSet):
    queryset = Character.objects.all()
    serializer_class = CharacterSerializer
    permission_classes = [AllowAny]  
    
    @action(detail=False, methods=['get'])
    def generate_characters(self, request):
        characters = self.get_queryset()
        return render(request, 'characters/character_list.html', {
            'characters': characters
        })

    @action(detail=True, methods=['post'])
    def generate_image(self, request, pk=None):
        try:
            character = self.get_object()
            generator = CharacterGenerator(character.story_id)
            
            # Only include fields that exist in your Character model
            character_data = {
                'name': character.name,
                'role': character.role,
                'background': character.background,
                'personality_traits': {
                    'general': character.background[:100]  # Using a snippet of background as traits
                }
            }
            
            image_path = generator.generate_character_image(character_data)
            
            # Update character with image path
            character.image_url = image_path
            character.save()
            
            return Response({
                'success': True,
                'image_url': image_path
            })
        except Exception as e:
            return Response({
                'success': False,
                'error': str(e)
            }, status=400)

class DialogueViewSet(viewsets.ModelViewSet):
    queryset = CharacterDialogue.objects.all()
    serializer_class = CharacterDialogueSerializer
    permission_classes = (AllowAny,)
    
    @action(detail=False, methods=['post'])
    def generate_dialogue(self, request):
        story_id = request.data.get('story_id')
        scene_description = request.data.get('scene_description')
        
        if not all([story_id, scene_description]):
            return Response(
                {"error": "story_id and scene_description are required"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
            
        generator = DialogueGenerator(story_id)
        dialogues = generator.generate_dialogue(scene_description)
        
        serializer = self.get_serializer(data=dialogues, many=True)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        
        return Response(serializer.data)

@require_http_methods(["POST"])
def generate_character_image(request, character_id):
    try:
        character = Character.objects.get(id=character_id)
        generator = CharacterGenerator(character.story_id)
        
        character_data = {
            'name': character.name,
            'role': character.role,
            'personality_traits': character.personality_traits,
            'physical_description': character.physical_description,
        }
        
        image_path = generator.generate_character_image(character_data)
        
        character.image_url = image_path
        character.save()
        
        return JsonResponse({
            'success': True,
            'image_url': image_path
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)
