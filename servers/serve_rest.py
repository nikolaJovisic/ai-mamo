import json
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from flask import Flask, request, send_file, send_from_directory, abort

import io
from flask_cors import CORS
from flask import jsonify
from matplotlib import cm

from ai.inference import infer as seg_infer
from ai.preprocess import negate_if_should
from mama.inference import infer as mama_infer
from mama.inference import get_encoder, get_head

from pydicom import dcmread

png_path = 'mammogram.png'
dcm_path = 'mammogram.dcm'

encoder = get_encoder('mama/mama_embed_pretrained_40k_steps_last_dinov2_vit_ckpt.pth')
head = get_head('mama/head_weights.pth')

app = Flask(__name__, static_folder='mamo-front/build/')
CORS(app)


with open(f'config.json', 'r') as config_file:
    config_data = json.load(config_file)

def dicom_analysis():
    try:
        dcm = dcmread(dcm_path)
    except Exception as e:
        return {"error": f"Invalid DICOM file: {str(e)}"}

    img = dcm.pixel_array.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = img.astype(np.uint8)
    Image.fromarray(np.stack([img]*3, axis=-1)).save(png_path)

    return {
        "name": dcm.get("PatientName", "Unknown"),
        "jmbg": dcm.get("PatientID", "Unknown"),
        "probability": mama_infer(encoder, head, dcm_path)
    }
    

def segment():
    image = Image.open(png_path)
    image = np.array(image)

    image = image[..., 0]
    image = negate_if_should(image)
    mask = seg_infer(image, binarize=False)


    plt.imsave('tmp.png', mask, cmap='gray')
    mask_rgb = plt.imread('tmp.png')

    rgb_image = np.stack([image, image, image], axis=-1)
    red_overlay = np.stack([mask_rgb[..., 0], np.zeros_like(mask_rgb[..., 0]), np.zeros_like(mask_rgb[..., 0])], axis=-1)
    blended_image = np.clip(rgb_image * (1 - red_overlay) + red_overlay * 255, 0, 255).astype(np.uint8)

    mask_report = (mask_rgb[..., :3] * 255).astype(np.uint8)

    plt.imsave('a.png', blended_image)
    plt.imsave('b.png', mask_report)


    report = np.concatenate([rgb_image, blended_image, mask_report], axis=1)

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
        file.save(dcm_path)

        payload = dicom_analysis()

        if "error" in payload:
            return jsonify(error=payload["error"]), 400
        
        heatmap = segment()

        heatmap.seek(0)
        response = send_file(
            path_or_file=heatmap,
            mimetype='image/png',
            as_attachment=True,
            download_name='processed_image.png'
        )
        response.headers['Name'] = payload['name']
        response.headers['JMBG'] = payload['jmbg']
        response.headers['Probability'] = payload['probability']
        return response




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
