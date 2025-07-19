from flask import Flask, request, jsonify
from flask_cors import CORS
from model.caption import generate_caption
from PIL import Image

app = Flask(__name__)

# 💣 FULLY open up CORS for all domains and methods
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})

@app.route('/caption', methods=['POST'])
def caption():
    try:
        if 'image' not in request.files:
            print("No image found in request.")
            return jsonify({'error': 'No image uploaded'}), 400

        model_name = request.form.get("model", "blip")
        image_file = request.files['image']
        image = Image.open(image_file.stream).convert("RGB")

        caption = generate_caption(image, model_name)
        return jsonify({'caption': caption})

    except Exception as e:
        import traceback
        print("❌ Error while generating caption:")
        traceback.print_exc()
        return jsonify({'error': 'Something went wrong on the server.'}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200
