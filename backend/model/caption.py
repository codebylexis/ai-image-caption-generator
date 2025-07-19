from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer
)
from PIL import Image
import torch

# ---- BLIP-2 Setup (Flan-T5-XL) ----
blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
blip2_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl",
    device_map="auto",
    torch_dtype=torch.float16
)

# ---- ViT+GPT2 Setup ----
vit_gpt2_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
vit_gpt2_feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
vit_gpt2_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_gpt2_model.to(device)

def generate_caption(image: Image.Image, model_name="blip") -> str:
    if model_name == "blip":
        # Use BLIP-2 (Flan-T5-XL)
        inputs = blip2_processor(image, return_tensors="pt").to(blip2_model.device)
        generated_ids = blip2_model.generate(**inputs, max_new_tokens=50)
        caption = blip2_processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return caption

    elif model_name == "vit_gpt2":
        # Use ViT + GPT2
        pixel_values = vit_gpt2_feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
        output_ids = vit_gpt2_model.generate(pixel_values, max_length=32)
        caption = vit_gpt2_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        return caption

    else:
        raise ValueError(f"Unknown model: {model_name}")
