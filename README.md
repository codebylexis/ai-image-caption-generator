# AI Image Caption Generator

A full-stack web app that generates natural language captions from images using modern transformer-based models like **BLIP-2 (Flan-T5-XL)** and **ViT-GPT2**.

## Features

- Upload image files (JPG/PNG)
- Generate captions using:
  - `BLIP-2` (`Salesforce/blip2-flan-t5-xl`)
  - `ViT + GPT2` (`nlpconnect/vit-gpt2-image-captioning`)
- Switch between models via dropdown
- Responsive frontend built with **React**
- Flask backend with **PyTorch** + **Transformers**

## Technologies

- Frontend: React, JavaScript, HTML/CSS
- Backend: Flask (Python), Transformers (HuggingFace), PyTorch
- Models: BLIP-2, ViT-GPT2
- Environment: Local execution with optional GPU support

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/codebylexis/ai-image-caption-generator.git
cd ai-image-caption-generator
```

### 2. Backend Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
python app.py
```

### 3. Frontend Setup

In another terminal:

```bash
cd frontend
npm install
npm start
```

Visit [http://localhost:3000](http://localhost:3000)

## Example Usage

1. Upload a `.jpg` or `.png` image
2. Select a model (e.g., ViT + GPT2 or BLIP-2)
3. Click **"Generate Caption"**
4. View AI-generated caption below the image

## File Structure

```
.
├── backend/
│   ├── app.py
│   ├── model/
│   │   └── caption.py
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── App.js
│       ├── index.js
│       └── App.css
```

## Notes

- ViT-GPT2 may take longer to generate captions (larger model).
- Model weights are downloaded on first run (ensure internet access).


## Future Work

- Add more advanced models (e.g., GIT, BLIP-2 OPT)
- Improve UI styling and feedback
- Deploy via Render/HuggingFace Spaces
- Save image/caption history
