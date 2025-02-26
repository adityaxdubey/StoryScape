from typing import List, Dict
# from google import genai
from ..models import Character, Story
import os
import json
import logging
import google.generativeai as genai
logger = logging.getLogger(__name__)

class CharacterGenerator:
    def __init__(self, story_id: int):
        """Initialize the character generator with story ID and Gemini API client."""
        self.story = Story.objects.get(id=story_id)
        # Initialize Gemini API client with the API key from settings
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            logger.error("GEMINI_API_KEY environment variable is not set")
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        else:
            logger.info(f"Using GEMINI_API_KEY: {api_key}")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')

    def generate_characters(self, plot: str) -> List[Dict]:
        """Generate characters based on the plot using Gemini API."""
        try:
            # Create the prompt for character generation
            prompt = self._create_character_prompt(plot)
            
            # Generate response using Gemini
            response = self.model.generate_content(prompt)
            
            # Parse the response
            characters_data = self._parse_character_response(response.text)
            return characters_data
            
        except Exception as e:
            logger.error(f"Error generating characters: {str(e)}")
            raise

    def _create_character_prompt(self, plot: str) -> str:
        """Create a detailed prompt for character generation."""
        return f"""
        Based on the following plot, create 3-4 detailed characters with rich personalities.
        Return the response in a JSON array format with each character having the following structure:
        
        Plot: {plot}
        
        Expected JSON format for each character:
        {{
            "name": "character name",
            "role": "protagonist/antagonist/supporting",
            "personality_traits": {{
                "trait1": "description",
                "trait2": "description"
            }},
            "background": "character's backstory",
            "goals": "character's objectives",
            "relationships": {{
                "character_name": "relationship description"
            }}
        }}
        
        Ensure the characters are complex and their motivations align with the plot.
        """

    def _parse_character_response(self, response_text: str) -> List[Dict]:
        """Parse the Gemini API response into structured character data."""
        try:
            # Clean up the response text to extract only the JSON part
            # Find the first '{' and last '}' to extract the JSON content
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("Invalid response format")
            
            json_str = response_text[start_idx:end_idx]
            characters_data = json.loads(json_str)
            
            # Validate the structure of each character
            validated_characters = []
            for char in characters_data:
                if self._validate_character_data(char):
                    validated_characters.append(char)
            
            return validated_characters
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {str(e)}")
            raise ValueError("Invalid JSON response from API")
        except Exception as e:
            logger.error(f"Error parsing character response: {str(e)}")
            raise

    def _validate_character_data(self, character: Dict) -> bool:
        """Validate the structure of character data."""
        required_fields = {
            'name': str,
            'role': str,
            'personality_traits': dict,
            'background': str,
            'goals': str,
            'relationships': dict
        }
        
        try:
            for field, field_type in required_fields.items():
                if field not in character:
                    logger.warning(f"Missing field '{field}' in character data")
                    return False
                if not isinstance(character[field], field_type):
                    logger.warning(f"Invalid type for field '{field}' in character data")
                    return False
            return True
        except Exception as e:
            logger.error(f"Error validating character data: {str(e)}")
            return False