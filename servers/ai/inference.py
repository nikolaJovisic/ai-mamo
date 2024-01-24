import os

import matplotlib.pyplot as plt
import numpy as np
import pydicom

from .postprocess import convert_soft_binary_to_hard_instance
from .preprocess import preprocess_scan, reverse_spatial_changes
from .segformer import (init_model, postprocess_segformer,
                        preprocess_for_segformer)

script_dir = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(script_dir, "model_weights")
model = init_model(weights_path)


def infer(image: np.ndarray, binarize: bool = True) -> np.ndarray:
    """
    Performs preprocessing, inference on the model and postprocessing for one image
    and returns instance mask of suspicious regions.
    :param binarize: binarize mask to hard instance mask
    :param image: in a grayscale format as in dicom pixel_array.
    :return: mask in a format of a matrix containing indices of each region of interest.
    """
    image, spatial_changes = preprocess_scan(image)
    image = preprocess_for_segformer(image)
    batch = np.expand_dims(image, axis=0)
    mask = model.predict(x=batch, batch_size=1).logits[0]
    mask = postprocess_segformer(mask)
    if binarize:
        mask = convert_soft_binary_to_hard_instance(mask)
    mask = reverse_spatial_changes(mask, spatial_changes)
    return mask


def demo():
    image = pydicom.dcmread("sample_dicom.dcm").pixel_array
    mask = infer(image)

    print(
        np.unique(mask)
    )  # for sample_dicom this should print [0, 1] as there is background and one suspicious region

    # uncomment to display this region

    # fig, ax = plt.subplots()
    # ax.imshow(image, cmap='gray')
    # mask_alpha = np.where(mask == 1, 0.5, 0)
    # ax.imshow(mask, cmap='jet', alpha=mask_alpha)
    # plt.show()


if __name__ == "__main__":
    demo()
