# core/serializers.py
from rest_framework import serializers
from .models import Story
from .services import StoryGenerationService

class StorySerializer(serializers.ModelSerializer):
    class Meta:
        model = Story
        fields = '__all__'

class StoryPremiseSerializer(serializers.Serializer):
    """
    Serializer for handling story premise input.
    Validates and processes the initial story information.
    """
    title = serializers.CharField(max_length=255, required=False)
    premise = serializers.CharField(required=True)
    
    def validate_premise(self, value):
        """
        Validates the story premise to ensure it's substantial enough
        for generating a meaningful story.
        """
        if len(value.split()) < 10:
            raise serializers.ValidationError(
                "Premise must be at least 10 words long to generate a meaningful story."
            )
        return value
    
    def create(self, validated_data):
        """
        Creates a new story instance, ensuring the author is set correctly.
        """
        service = StoryGenerationService()
        analysis = service.analyze_premise(validated_data['premise'])
        generated_story = service.generate_expanded_plot(validated_data['premise'], analysis)
        story = Story.objects.create(
            title=validated_data.get('title', 'Untitled Story'),
            author=self.context['request'].user,
            basic_premise=validated_data['premise'],
            expanded_plot=generated_story
        )
        return story 