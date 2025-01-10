# story_generator/views.py
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from .models import Story
from .serializers import StorySerializer, StoryPremiseSerializer
# from .services import get_story_generator
from .services import get_story_generator
import logging
from rest_framework.viewsets import ViewSet
from rest_framework.permissions import AllowAny
logger = logging.getLogger(__name__)


class StoryViewSet(viewsets.ViewSet):
    """
    ViewSet for handling story-related operations.
    Provides endpoints for story creation, generation, and modification.
    """
    queryset = Story.objects.all()
    serializer_class = StorySerializer
    permission_classes = [AllowAny]
    
    @action(detail=False, methods=['post'])
    def generate(self, request):
        """
        Generates a new story based on a provided premise.
        Handles the story generation process and saves the result.
        """
        # Validate input
        serializer = StoryPremiseSerializer(data=request.data, context={'request': request})
        if serializer.is_valid():
            story = serializer.save()
            response_serializer = StorySerializer(story)
            return Response(response_serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
       
        try:
            # Get the story generator service
            generator = get_story_generator()
            
            # Analyze the premise
            premise = serializer.validated_data['premise']
            analysis = generator.analyze_premise(premise)
            
            # Generate the expanded plot
            expanded_plot = generator.generate_expanded_plot(premise, analysis)
            
            # Create and save the story
            story = Story.objects.create(
                title=serializer.validated_data.get('title', 'Untitled Story'),
                author=request.user,
                basic_premise=premise,
                expanded_plot=expanded_plot
            )
            
            # Return the created story
            response_serializer = StorySerializer(story)
            return Response(response_serializer.data, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            logger.error(f"Error generating story: {str(e)}")
            return Response(
                {"error": "Failed to generate story"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    @action(detail=False, methods=['post'])
    def generate_story(self, request):
        """
        Generates a new story based on a provided premise.
        Handles the story generation process and saves the result.
        """
        # Validate input
        serializer = StoryPremiseSerializer(data=request.data, context={'request': request})
        if serializer.is_valid():
            premise = serializer.validated_data['premise']
            title = serializer.validated_data.get('title', 'Untitled Story')
            
            try:
                # Get the story generator service
                generator = get_story_generator()
                
                # Analyze the premise
                analysis = generator.analyze_premise(premise)
                
                # Generate the expanded plot
                expanded_plot = generator.generate_expanded_plot(premise, analysis)
                
                # Create and save the story
                story = Story.objects.create(
                    title=title,
                    author=request.user,
                    basic_premise=premise,
                    expanded_plot=expanded_plot
                )
                
                # Return the created story
                response_serializer = StorySerializer(story)
                return Response(response_serializer.data, status=status.HTTP_201_CREATED)
                
            except Exception as e:
                logger.error(f"Error generating story: {str(e)}")
                return Response(
                    {"error": "Failed to generate story"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    
    @action(detail=True, methods=['post'])
    def regenerate_segment(self, request, pk=None):
        """
        Regenerates a specific segment of an existing story.
        Useful for iterating on parts of the story that need improvement.
        """
        story = self.get_object()
        segment_type = request.data.get('segment_type')
        
        if segment_type not in ['setup', 'conflict', 'development', 'resolution']:
            return Response(
                {"error": "Invalid segment type"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            generator = get_story_generator()
            
            # Get the original analysis
            analysis = generator.analyze_premise(story.basic_premise)
            
            # Regenerate the specified segment
            new_segment = generator._generate_story_segment(
                generator._create_story_prompt(story.basic_premise, analysis),
                segment_type,
                analysis
            )
            
            # Update the story's plot structure
            expanded_plot = story.expanded_plot
            expanded_plot[segment_type] = new_segment
            story.expanded_plot = expanded_plot
            story.save()
            
            return Response(StorySerializer(story).data)
            
        except Exception as e:
            logger.error(f"Error regenerating segment: {str(e)}")
            return Response(
                {"error": "Failed to regenerate segment"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
from django.shortcuts import render

def index(request):
    return render(request, 'core/index.html')

from django.shortcuts import render, get_object_or_404, redirect
from .models import Story
from .forms import StoryForm

def story_list(request):
    stories = Story.objects.all()
    return render(request, 'core/story_list.html', {'stories': stories})

def story_detail(request, pk):
    story = get_object_or_404(Story, pk=pk)
    return render(request, 'core/story_detail.html', {'story': story})

from django.views.generic.edit import CreateView
from .models import Story
from .forms import StoryForm

class StoryCreateView(CreateView):
    model = Story
    form_class = StoryForm
    template_name = 'core/story_form.html'  # Specify the template for the form
    success_url = 'core/story_list'  # Specify the URL to redirect to after form submission

def story_update(request, pk):
    story = get_object_or_404(Story, pk=pk)
    if request.method == 'POST':
        form = StoryForm(request.POST, instance=story)
        if form.is_valid():
            form.save()
            return redirect('story_list')
    else:
        form = StoryForm(instance=story)
    return render(request, 'core/story_form.html', {'form': form})

def story_delete(request, pk):
    story = get_object_or_404(Story, pk=pk)
    if request.method == 'POST':
        story.delete()
        return redirect('story_list')
    return render(request, 'core/story_confirm_delete.html', {'story': story})
