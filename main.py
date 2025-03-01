!pip install googletrans==4.0.0-rc1
!pip install transformers
!pip install gTTS

#Importing the Model
from googletrans import Translator
from transformers import pipeline

translator = Translator()
generator = pipeline('text-generation', model='t5-base')

#Translating English Commentary to Tamil
def generate_tamil_commentary(commentary_text):
    # Translate English commentary to Tamil
    translated_text = translator.translate(commentary_text, dest='ta').text

    # Use the text generation pipeline to refine the translated commentary
    generated_commentary = generator(translated_text, max_length=100, num_return_sequences=1)[0]['generated_text']
    return generated_commentary

#Generated Tamil Commentary
english_commentary = "Virat Kholi hits a massive six out of the ground!"
tamil_commentary = generate_tamil_commentary(english_commentary)
print(tamil_commentary)
