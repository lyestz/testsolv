# api/index.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
import base64
from io import BytesIO
from PIL import Image
import torch
from torchvision import transforms as T
from log import print_green_ascii_art
import psutil
import traceback
import contextlib
import io
import ssl

# Optional: Disable SSL verification for dev
ssl._create_default_https_context = ssl._create_unverified_context

logging.basicConfig(level=logging.WARNING)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# ===== Silent torch hub load =====
def silent_torch_hub_load(*args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return torch.hub.load(*args, **kwargs)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)
CORS(app)

class TextRecognitionApp:
    models = ['vitstr']

    def __init__(self):
        self.device = device
        self._model_cache = {}
        self._preprocess = T.Compose([
            T.Resize((32, 128), T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])
        self.preload_models()

    def preload_models(self):
        for model_name in self.models:
            try:
                self._get_model(model_name)
            except Exception as e:
                logging.error(f"Failed to preload model {model_name}: {e}")

    def _get_model(self, name):
        if name not in self.models:
            raise ValueError(f"Model '{name}' is not supported.")
        if name not in self._model_cache:
            try:
                model = silent_torch_hub_load('baudm/parseq', name, pretrained=True).eval()
            except Exception as e:
                logging.warning(f"Online model load failed: {e}")
                model = silent_torch_hub_load('models/parseq', name, pretrained=True, source='local').eval()
            model.to(self.device)
            self._model_cache[name] = model
        return self._model_cache[name]

    @torch.inference_mode()
    def process_image(self, model_name, image_base64):
        try:
            if not image_base64:
                return "", False
            image_data = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_data)).convert('RGB')
            image = self._preprocess(image).unsqueeze(0).to(self.device)
            model = self._get_model(model_name)
            pred = model(image).softmax(-1)
            label, _ = model.tokenizer.decode(pred)
            del image
            del pred
            torch.cuda.empty_cache()
            return label[0], True
        except Exception as e:
            logging.error(f"Error processing image: {traceback.format_exc()}")
            return str(e), False

    def log_memory_usage(self):
        process = psutil.Process(os.getpid())
        logging.info(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

recognition_app = TextRecognitionApp()

@app.route('/', methods=['GET', 'POST'])
def handle_request():
    model_name = request.args.get('a', 'vitstr') 
    image_base64 = request.args.get('b', '')
    number_to_compare = request.args.get('n', '')
    recognition_app.log_memory_usage()
    try:
        recognized_text, valid = recognition_app.process_image(model_name, image_base64)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    if not valid:
        return jsonify({"status": "error", "message": "Invalid image data"})
    elif recognized_text == number_to_compare:
        return jsonify({"status": "ok", "message": f"Image {number_to_compare} solved"})
    else:
        return jsonify({"status": "not ok", "message": f"Image {number_to_compare} does not match"})

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "alive"})

# Export app for Vercel
handler = app
