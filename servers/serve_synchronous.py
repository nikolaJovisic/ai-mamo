import numpy as np
from PIL import Image
from flask import Flask, request, send_file
import os

from matplotlib import pyplot as plt
from werkzeug.utils import secure_filename
import io
from flask_cors import CORS
from ai.inference import infer

app = Flask(__name__)
CORS(app)

def process(image_stream):
    image = Image.open(image_stream)
    image = np.array(image)

    image = image[..., 0]
    mask = infer(image, binarize=False)

    rgb_image = np.stack([image, image, image], axis=-1)
    red_overlay = np.stack([mask, np.zeros_like(mask), np.zeros_like(mask)], axis=-1)
    blended_image = np.clip(rgb_image * (1 - red_overlay) + red_overlay * 255, 0, 255).astype(np.uint8)

    rgb_mask = (np.stack([mask, mask, mask], axis=-1) * 255).astype(np.uint8)
    report = np.concatenate([blended_image, rgb_mask], axis=1)

    print(image)
    print(np.unique(mask))

    processed_image = Image.fromarray(report)
    processed_image_stream = io.BytesIO()
    processed_image.save(processed_image_stream, format='PNG')
    return processed_image_stream

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if file:
        filename = secure_filename(file.filename)
        # Convert the file to a BytesIO object for processing
        file_stream = io.BytesIO()
        file.save(file_stream)
        file_stream.seek(0)

        # Process the image
        processed_image = process(file_stream)

        # Send back the processed image
        processed_image.seek(0)
        return send_file(
            path_or_file=processed_image, # This should be the path to your file or a file object
            mimetype='image/png',
            as_attachment=True,
            download_name='processed_image.png' # Use the appropriate file name here
        )

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
