from django.db import models
from core.models import Story

class Character(models.Model):
    story = models.ForeignKey(Story, on_delete=models.CASCADE, related_name='characters')
    name = models.CharField(max_length=100)
    role = models.CharField(max_length=100)  # protagonist, antagonist, supporting
    personality_traits = models.JSONField()  # Store personality traits as JSON
    background = models.TextField()
    physical_description = models.TextField(blank=True, null=True)
    image_url = models.CharField(max_length=255, blank=True, null=True)
    goals = models.TextField()
    relationships = models.JSONField()  # Store relationships with other characters
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} - {self.role} in {self.story.title}"

class CharacterDialogue(models.Model):
    character = models.ForeignKey(Character, on_delete=models.CASCADE, related_name='dialogues')
    scene_number = models.IntegerField()
    dialogue_text = models.TextField()
    emotion = models.CharField(max_length=50)
    sequence_number = models.IntegerField()
    
    class Meta:
        ordering = ['scene_number', 'sequence_number']
