import json
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from flask import Flask, request, send_file, send_from_directory, abort

import io
from flask_cors import CORS
from matplotlib import cm

from ai.inference import infer

app = Flask(__name__, static_folder='mamo-front/build/')
CORS(app)

with open(f'config.json', 'r') as config_file:
    config_data = json.load(config_file)


def process(image_stream):
    image = Image.open(image_stream)
    image = np.array(image)

    image = image[..., 0]
    mask = infer(image, binarize=False)

    plt.imsave('tmp.png', mask, cmap='gray')
    mask_rgb = plt.imread('tmp.png')

    rgb_image = np.stack([image, image, image], axis=-1)
    red_overlay = np.stack([mask_rgb[..., 0], np.zeros_like(mask_rgb[..., 0]), np.zeros_like(mask_rgb[..., 0])], axis=-1)
    blended_image = np.clip(rgb_image * (1 - red_overlay) + red_overlay * 255, 0, 255).astype(np.uint8)

    mask_report = (mask_rgb[..., :3] * 255).astype(np.uint8)

    plt.imsave('a.png', blended_image)
    plt.imsave('b.png', mask_report)


    report = np.concatenate([blended_image, mask_report], axis=1)

    processed_image = Image.fromarray(report)
    processed_image_stream = io.BytesIO()
    processed_image.save(processed_image_stream, format='PNG')
    return processed_image_stream


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    if path != "" and app.static_folder:
        return app.send_static_file(path)
    return app.send_static_file(r'index.html')


@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file:
        file_stream = io.BytesIO()
        file.save(file_stream)
        file_stream.seek(0)

        processed_image = process(file_stream)

        processed_image.seek(0)
        return send_file(
            path_or_file=processed_image,
            mimetype='image/png',
            as_attachment=True,
            download_name='processed_image.png'
        )


@app.route("/<id>", methods=["GET"])
def download_file(id):
    directory = os.path.join(config_data['data_path'], id)
    filename = f"{id}.png"
    if os.path.isfile(os.path.join(directory, filename)):
        return send_from_directory(directory, filename, as_attachment=True)
    else:
        abort(404)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=config_data['internal_port'], debug=True)
