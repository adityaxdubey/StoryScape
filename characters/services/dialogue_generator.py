from typing import List, Dict
from google import genai
from ..models import Character, CharacterDialogue
import os
import logging

logger = logging.getLogger(__name__)

class DialogueGenerator:
    def __init__(self, story_id: int):
        self.characters = Character.objects.filter(story_id=story_id)
        # Initialize Gemini API client
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')

    def generate_dialogue(self, scene_description: str) -> List[Dict]:
        """
        Generate dialogue between characters for a specific scene
        """
        try:
            prompt = self._create_dialogue_prompt(scene_description)
            
            # Generate response using Gemini
            response = self.model.generate_content(prompt)
            
            # Parse the response
            dialogues = self._parse_dialogue_response(response.text)
            return dialogues
            
        except Exception as e:
            logger.error(f"Error generating dialogue: {str(e)}")
            raise

    def _create_dialogue_prompt(self, scene_description: str) -> str:
        characters_info = "\n".join([
            f"- {char.name}: {char.personality_traits}" 
            for char in self.characters
        ])
        
        return f"""
        Create natural dialogue for the following scene with these characters:
        
        Characters:
        {characters_info}
        
        Scene Description:
        {scene_description}
        
        Generate a sequence of dialogue that:
        1. Reflects each character's personality
        2. Advances the plot
        3. Shows character relationships
        4. Includes emotions and reactions
        """

    def _parse_dialogue_response(self, response_text: str) -> List[Dict]:
        # Implement parsing logic here
        # Convert the AI response into structured dialogue data
        # Example: Split the response into lines and parse each line into a dialogue entry
        dialogues = []
        for line in response_text.splitlines():
            if line.strip():
                # Example parsing logic, adjust according to your response format
                character_name, dialogue_text = line.split(":", 1)
                dialogues.append({
                    "character": character_name.strip(),
                    "dialogue": dialogue_text.strip()
                })
        return dialogues 