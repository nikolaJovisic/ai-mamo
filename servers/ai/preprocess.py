import cv2
import matplotlib.pyplot as plt
import numpy as np


def binarize(img):
    img = img.astype(np.uint8)

    _, mask = cv2.threshold(src=img, thresh=0.1, maxval=1, type=cv2.THRESH_BINARY)

    return mask


def dilate(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    return dilated_mask


def keep_largest_blob(mask):
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a mask with only the largest contour
    largest_blob_mask = np.zeros_like(mask)
    cv2.drawContours(largest_blob_mask, [largest_contour], -1, 1, thickness=cv2.FILLED)

    return largest_blob_mask


def get_breast_mask(image):
    mask = binarize(image)
    mask = dilate(mask)
    return keep_largest_blob(mask)


def keep_only_breast(image):
    mask = get_breast_mask(image)
    return image * mask, mask


def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=5)
    return clahe.apply(image)


def crop_borders(image):
    l = 0.01
    r = 0.01
    u = 0.04
    d = 0.04

    n_rows, n_cols = image.shape

    l_crop = int(n_cols * l)
    r_crop = int(n_cols * (1 - r))
    u_crop = int(n_rows * u)
    d_crop = int(n_rows * (1 - d))

    cropped_img = image[u_crop:d_crop, l_crop:r_crop]

    return cropped_img, (l_crop, n_cols - r_crop, u_crop, n_rows - d_crop)


def should_flip(image):
    x_center = image.shape[1] // 2
    col_sum = image.sum(axis=0)

    left_sum = sum(col_sum[0:x_center])
    right_sum = sum(col_sum[x_center:-1])

    return left_sum < right_sum


def pad(image):
    n_rows, n_cols = image.shape
    if n_rows == n_cols:
        return image

    target_shape = (np.maximum(n_rows, n_cols),)*2

    padded_img = np.zeros(shape=target_shape).astype(image.dtype)
    padded_img[:n_rows, :n_cols] = image

    return padded_img


def negate_if_should(image):
    threshold = 0.2 * np.max(image)
    return image if np.mean(image > threshold) < 0.5 else np.max(image) - image


def preprocess_scan_with_mask(image, mass_mask):
    image = negate_if_should(image)
    image, borders = crop_borders(image)
    mass_mask, _ = crop_borders(mass_mask)
    image, breast_mask = keep_only_breast(image)
    flip = should_flip(image)

    if flip:
        image = np.fliplr(image)
        breast_mask = np.fliplr(breast_mask)
        mass_mask = np.fliplr(mass_mask)

    image = apply_clahe(image)
    image = image * breast_mask  # clahe manages to change black to slight gray
    shape_before_padding = image.shape
    image = pad(image)
    mass_mask = pad(mass_mask)

    spatial_changes = (borders, flip, shape_before_padding, image.shape)

    return image, mass_mask, spatial_changes


def preprocess_scan(image):
    image, _, spatial_changes = preprocess_scan_with_mask(image, image)
    return image, spatial_changes


def reverse_spatial_changes(image, spatial_changes):
    borders, flip, shape_before_padding, shape = spatial_changes
    l, r, u, d = borders
    image = cv2.resize(src=image, dsize=shape, interpolation=cv2.INTER_NEAREST)
    image = image[: shape_before_padding[0], : shape_before_padding[1]]
    if flip:
        image = np.fliplr(image)
    image = np.pad(image, ((u, d), (l, r)))
    return image
