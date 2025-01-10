# story_generator/services.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import spacy
import torch
from textblob import TextBlob
from typing import Dict, List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StoryGenerationService:
    def __init__(self):
        """
        Initialize the story generation service with necessary models and tools.
        We use GPT-2 for creative text generation and spaCy for text analysis.
        """
        try:
            # Initialize language models
            self.nlp = spacy.load("en_core_web_sm")
            self.generator_model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
            
            # Set up padding token
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.generator_model.config.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.add_special_tokens({
            'pad_token': '[PAD]'})
            
            # Set device (GPU if available)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.generator_model.to(self.device)
            
            # Set up story structure templates
            self.story_elements = {
                "setup": ["context", "main_character", "initial_situation"],
                "conflict": ["challenge", "obstacle", "antagonist"],
                "development": ["complications", "character_growth", "subplots"],
                "resolution": ["climax", "outcome", "conclusion"],
            }
            
            logger.info("Story Generation Service initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Story Generation Service: {str(e)}")
            raise RuntimeError(f"Failed to initialize service: {str(e)}")

    def _analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of the given text using TextBlob.
        Returns sentiment polarity (-1 to 1).
        """
        try:
            analysis = TextBlob(text)
            return analysis.sentiment.polarity
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return 0.0

    def _extract_key_phrases(self, text: str) -> List[str]:
        """
        Extract key phrases (noun chunks) from the given text using spaCy.
        """
        try:
            doc = self.nlp(text)
            return [chunk.text for chunk in doc.noun_chunks]
        except Exception as e:
            logger.error(f"Error extracting key phrases: {str(e)}")
            return []

    def _evaluate_story_potential(self, analysis: Dict) -> str:
        """
        Evaluate the story potential based on sentiment and key phrases.
        Returns a qualitative assessment as a string.
        """
        try:
            sentiment = analysis.get("sentiment", 0)
            key_phrases = analysis.get("key_phrases", [])

            if sentiment > 0.5 and len(key_phrases) > 3:
                return "High potential"
            elif sentiment > 0 and len(key_phrases) > 1:
                return "Moderate potential"
            else:
                return "Low potential"
        except Exception as e:
            logger.error(f"Error evaluating story potential: {str(e)}")
            return "Low potential"

    def _extract_entities(self, text: str) -> Dict:
        """
        Extract and categorize named entities from the text using spaCy.
        """
        try:
            doc = self.nlp(text)
            entities = {
                "characters": [],
                "locations": [],
                "organizations": [],
                "misc": [],
            }

            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    entities["characters"].append(ent.text)
                elif ent.label_ in ["GPE", "LOC"]:
                    entities["locations"].append(ent.text)
                elif ent.label_ == "ORG":
                    entities["organizations"].append(ent.text)
                else:
                    entities["misc"].append((ent.text, ent.label_))

            return entities
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return {
                "characters": [],
                "locations": [],
                "organizations": [],
                "misc": [],
            }

    def _identify_themes(self, text: str) -> List[str]:
        """
        Identify potential themes in the story based on key words and phrases.
        """
        try:
            doc = self.nlp(text)
            themes = []
            theme_keywords = {
                "love": ["romance", "relationship", "heart", "passion"],
                "redemption": ["forgiveness", "change", "growth", "overcome"],
                "conflict": ["struggle", "fight", "battle", "challenge"],
                "mystery": ["secret", "puzzle", "unknown", "discover"],
                "adventure": ["journey", "quest", "explore", "discover"],
                "friendship": ["friend", "companion", "loyalty", "trust"],
            }

            text_lower = doc.text.lower()
            for theme, keywords in theme_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    themes.append(theme)

            return themes if themes else ["general"]
        except Exception as e:
            logger.error(f"Error identifying themes: {str(e)}")
            return ["general"]

    def analyze_premise(self, premise: str) -> Dict:
        """
        Analyzes the basic premise to extract key story elements and themes.
        """
        try:
            if not premise or not isinstance(premise, str):
                raise ValueError("Invalid premise provided")

            analysis = {
                'entities': self._extract_entities(premise),
                'themes': self._identify_themes(premise),
                'sentiment': self._analyze_sentiment(premise),
                'key_phrases': self._extract_key_phrases(premise),
            }
            
            # Ensure themes exist
            if not analysis.get('themes'):
                analysis['themes'] = ['general']
                
            story_potential = self._evaluate_story_potential(analysis)

            logger.debug(f"Analysis result: {analysis}")
            
            return {
                'analysis': analysis,
                'potential_directions': story_potential,
            }
        except Exception as e:
            logger.error(f"Error in analyze_premise: {str(e)}")
            return {
                'analysis': {
                    'entities': {'characters': [], 'locations': [], 'organizations': [], 'misc': []},
                    'themes': ['general'],
                    'sentiment': 0.0,
                    'key_phrases': [],
                },
                'potential_directions': 'Low potential'
            }

    def _create_story_prompt(self, premise: str, analysis: Dict) -> str:
        """
        Creates a detailed prompt for the story generation model.
        """
        try:
            # Safely get values with fallbacks
            themes = analysis.get('analysis', {}).get('themes', ['general'])
            entities = analysis.get('analysis', {}).get('entities', {})
            characters = entities.get('characters', [])
            locations = entities.get('locations', [])

            prompt_elements = [
                f"Premise: {premise}",
                f"Themes: {', '.join(themes)}",
                f"Characters: {', '.join(characters)}",
                f"Setting: {', '.join(locations)}",
            ]
            return "\n".join(prompt_elements)
        except Exception as e:
            logger.error(f"Error creating story prompt: {str(e)}")
            return f"Premise: {premise}\nThemes: general"

    def _generate_story_segment(self, prompt: str, segment_type: str, analysis: Dict) -> str:
            """
            Generates a specific segment of the story (setup, conflict, etc.).
            Now includes proper attention mask handling.
            """
            try:
                segment_prompt = f"{prompt}\nFocus on {segment_type} elements: {', '.join(self.story_elements[segment_type])}"

                # Ensure the model is in evaluation mode
                self.generator_model.eval()

                # Tokenize with proper padding and attention mask
                encoded_input = self.tokenizer(
                    segment_prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024
                )
                
                # Move everything to the correct device
                input_ids = encoded_input["input_ids"].to(self.device)
                attention_mask = encoded_input["attention_mask"].to(self.device)

                with torch.no_grad():
                    outputs = self.generator_model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_length=150,
                        num_return_sequences=1,
                        no_repeat_ngram_size=2,
                        temperature=0.8,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id,
                        do_sample=True,
                        # Set EOS token ID to ensure proper sequence termination
                        eos_token_id=self.tokenizer.eos_token_id,
                        # Add bos token id if not already present in input
                        bos_token_id=self.tokenizer.bos_token_id,
                    )

                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Clean up the generated text by removing the prompt
                generated_text = generated_text.replace(segment_prompt, '').strip()
                
                return generated_text
            except Exception as e:
                logger.error(f"Error generating story segment {segment_type}: {str(e)}")
                return f"Could not generate {segment_type} segment due to an error."

    def generate_expanded_plot(self, premise: str, analysis: Dict) -> Dict:
        """
        Generates a detailed plot structure from the basic premise.
        """
        try:
            if not premise or not isinstance(premise, str):
                raise ValueError("Invalid premise provided")

            prompt = self._create_story_prompt(premise, analysis)
            
            plot_structure = {}
            for segment in ['setup', 'conflict', 'development', 'resolution']:
                try:
                    plot_structure[segment] = self._generate_story_segment(
                        prompt, segment, analysis
                    )
                except Exception as e:
                    logger.error(f"Error generating {segment}: {str(e)}")
                    plot_structure[segment] = f"Could not generate {segment}"

            return plot_structure
        except Exception as e:
            logger.error(f"Error in generate_expanded_plot: {str(e)}")
            raise RuntimeError(f"Failed to generate plot: {str(e)}")

def get_story_generator():
    """
    Factory function to create and return a StoryGenerationService instance.
    Implements singleton pattern to avoid multiple model loads.
    """
    try:
        if not hasattr(get_story_generator, "instance"):
            get_story_generator.instance = StoryGenerationService()
        return get_story_generator.instance
    except Exception as e:
        logger.error(f"Error in get_story_generator: {str(e)}")
        raise RuntimeError(f"Failed to get story generator instance: {str(e)}")
    

    
