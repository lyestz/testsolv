from PIL import Image
import torch
import io
import base64

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model and transform only once
print("🔄 Loading model...")
model = torch.hub.load('hzwer/arcnn', 'vitstr_base_patch16_224', pretrained=True)
model.eval()
transform = torch.hub.load('hzwer/arcnn', 'vitstr_transform')
print("✅ Model loaded.")

def handler(request):
    if request.method == "GET":
        return jsonify({"message": "✅ CAPTCHA Solver is running with ViTSTR."})

    if request.method == "POST":
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        try:
            img_file = request.files['image']
            image = Image.open(img_file.stream).convert('RGB')

            img_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                pred = model(img_tensor)
                result = pred[0]

            return jsonify({"code": result})

        except Exception as e:
            return jsonify({"error": str(e)}), 500
