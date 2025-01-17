from rest_framework import serializers
from .models import Character, CharacterDialogue

class CharacterSerializer(serializers.ModelSerializer):
    class Meta:
        model = Character
        fields = '__all__'

class CharacterDialogueSerializer(serializers.ModelSerializer):
    class Meta:
        model = CharacterDialogue
        fields = '__all__'
