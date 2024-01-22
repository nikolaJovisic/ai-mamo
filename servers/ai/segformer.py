import os

import cv2
import numpy as np
from transformers import TFSegformerForSemanticSegmentation
import tensorflow as tf
from tensorflow import keras

mean = tf.constant([0.485, 0.456, 0.406])
std = tf.constant([0.229, 0.224, 0.225])


def normalize(image):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = (image - mean) / tf.maximum(std, keras.backend.epsilon())
    return image


def preprocess_for_segformer(image):
    image = cv2.resize(src=image, dsize=(1024,) * 2, interpolation=cv2.INTER_LANCZOS4)

    image = np.expand_dims(image, -1).repeat(3, axis=-1)
    image[..., 1] = 1 - image[..., 1]
    image = normalize(image)
    image = tf.transpose(image, (2, 0, 1))
    return image


def init_model(weights_path):
    model_checkpoint = os.path.join(weights_path, 'model')
    initial_model = TFSegformerForSemanticSegmentation.from_pretrained(
        model_checkpoint,
        num_labels=1,
        ignore_mismatched_sizes=True,
        hidden_dropout_prob=0.3,
        attention_probs_dropout_prob=0.3,
        classifier_dropout_prob=0.3,
    )

    latest = tf.train.latest_checkpoint(weights_path)
    initial_model.load_weights(filepath=latest)
    return initial_model


def postprocess_segformer(mask):
    mask = tf.sigmoid(mask)
    mask = tf.transpose(mask, (1, 2, 0))
    mask = tf.image.resize(mask, [1024, 1024])
    mask = tf.squeeze(mask)
    return mask.numpy()
