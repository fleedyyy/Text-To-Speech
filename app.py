import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import gradio as gr
from scipy.io.wavfile import write
import tempfile
import re

# pakai GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model yang sudah di train
model = SpeechT5ForTextToSpeech.from_pretrained("Marcent/SpeechT5_finetune_TTS")
processor = SpeechT5Processor.from_pretrained("Marcent/SpeechT5_finetune_TTS")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Load dataset untuk speaker embedding
dataset = load_dataset("Marcent/tts_dataset")
example = dataset["test"][304]
speaker_embeddings = torch.tensor(example["speaker_embeddings"]).unsqueeze(0)

model = model.to(device)
vocoder = vocoder.to(device)


def tts_fn(text):

    number_words = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
    10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
    18: "eighteen", 19: "nineteen", 20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy",
    80: "eighty", 90: "ninety", 100: "one hundred", 1000: "one thousand"
    }

    def number_to_words(number):
        if number < 20:
            return number_words[number]
        elif number < 100:
            tens, unit = divmod(number, 10)
            return number_words[tens * 10] + (" " + number_words[unit] if unit else "")
        elif number < 1000:
            hundreds, remainder = divmod(number, 100)
            return (number_words[hundreds] + " hundreds" if hundreds > 1 else "hundred") + (" " + number_to_words(remainder) if remainder else "")
        elif number < 1000000:
            thousands, remainder = divmod(number, 1000)
            return (number_to_words(thousands) + " thousands" if thousands > 1 else "thousand") + (" " + number_to_words(remainder) if remainder else "")
        elif number < 1000000000:
            millions, remainder = divmod(number, 1000000)
            return number_to_words(millions) + " millions" + (" " + number_to_words(remainder) if remainder else "")
        elif number < 1000000000000:
            billions, remainder = divmod(number, 1000000000)
            return number_to_words(billions) + " billions" + (" " + number_to_words(remainder) if remainder else "")
        else:
            return str(number)
    
    # Replace nomer dengan words
    def replace_numbers_with_words(text):
        def replace(match):
            number = int(match.group())
            return number_to_words(number)

        # Find the numbers and change with words.
        result = re.sub(r'\b\d+\b', replace, text)

        return result

    replacements = [
        ("â", "a"),  # Long a
        ("ç", "ch"),  # Ch as in "chair"
        ("ğ", "gh"),  # Silent g or slight elongation of the preceding vowel
        ("ı", "i"),   # Dotless i
        ("î", "i"),   # Long i
        ("ö", "oe"),  # Similar to German ö
        ("ş", "sh"),  # Sh as in "shoe"
        ("ü", "ue"),  # Similar to German ü
        ("û", "u"),   # Long u
    ]

    def normalize_text(text):
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation (except apostrophes)
        text = re.sub(r'[^\w\s\']', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def cleanup_text(text):
        for src, dst in replacements:
            text = text.replace(src, dst)
        return text


    converted_text = replace_numbers_with_words(text)
    cleaned_text = cleanup_text(converted_text)
    final_text = normalize_text(cleaned_text)    
    inputs = processor(text=final_text, return_tensors="pt").to(device)

    # Add speaker embedding
    with torch.no_grad():
        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings.to(device), vocoder=vocoder)

    # temp wav
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        write(fp.name, rate=16000, data=speech.cpu().numpy())
        return fp.name

# Gradio 
gr.Interface(
    fn=tts_fn,
    inputs=gr.Textbox(label="Enter text"),
    outputs=gr.Audio(label="Audio"),
    title="SpeechT5 TTS",
    description="Fine-tuned SpeechT5 TTS",
).launch(share=True)

