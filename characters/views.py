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
logger = logging.getLogger(__name__)

class CharacterViewSet(viewsets.ModelViewSet):
    queryset = Character.objects.all()
    serializer_class = CharacterSerializer
    permission_classes = [AllowAny]  
    
    @action(detail=False, methods=['post'], url_path='generate-characters', permission_classes=[AllowAny])
    def generate_characters(self, request):
        story_id = request.data.get('story_id')
        plot = request.data.get('plot')
        
        if not all([story_id, plot]):
            return Response(
                {"error": "story_id and plot are required"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            generator = CharacterGenerator(story_id)
            characters_data = generator.generate_characters(plot)
            
            # Add the story to each character
            for character_data in characters_data:
                character_data['story'] = story_id
            
            # Save characters to database
            serializer = CharacterSerializer(data=characters_data, many=True)
            serializer.is_valid(raise_exception=True)
            serializer.save()
            
            return Response(serializer.data)
            
        except Story.DoesNotExist:
            return Response(
                {"error": f"Story with id {story_id} does not exist"},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            logger.error(f"Error generating characters: {str(e)}")
            return Response(
                {"error": "Failed to generate characters"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

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

def character_list(request):
    characters = Character.objects.all()
    return render(request, 'characters/character_list.html', {
        'characters': characters
    })
