def generate_character_image_view(request, character_data):
    generator = CharacterGenerator(story_id=story_id)
    image_path = generator.generate_character_image(character_data)
    return JsonResponse({'image_url': image_path}) 