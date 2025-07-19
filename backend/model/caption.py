from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load BLIP once at startup
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image: Image.Image, model_name="blip") -> str:
    if model_name == "blip":
        inputs = blip_processor(image, return_tensors="pt")
        out = blip_model.generate(**inputs)
        return blip_processor.decode(out[0], skip_special_tokens=True)

    elif model_name == "vit_gpt2":
        # Stub for future support
        return "ViT+GPT2 model support coming soon."

    else:
        raise ValueError(f"Unknown model: {model_name}")
