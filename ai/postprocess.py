from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu


def convert_soft_binary_to_hard_instance(soft_mask):
    soft_mask = cv2.GaussianBlur(soft_mask, (5, 5), 0)
    otsu_tr = threshold_otsu(soft_mask) * 0.9
    binary_mask = np.where(soft_mask >= otsu_tr, 1, 0).astype(np.uint8)

    kernel = np.ones((50, 50), np.uint8)
    binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    binary_mask = cv2.erode(binary_mask, kernel, iterations=1)

    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    instance_mask = np.zeros_like(soft_mask, dtype=np.uint8)

    for i, contour in enumerate(contours):
        cv2.drawContours(instance_mask, [contour], -1, (i + 1), -1)

    return instance_mask


def display_polygons(image, *polygons):
    plt.figure()

    if image is not None:
        plt.imshow(image)

    for polygon in polygons:
        coordinates = np.array(polygon)
        coordinates = np.squeeze(coordinates)
        coordinates = np.vstack((coordinates, coordinates[0]))

        x = coordinates[:, 0]
        y = coordinates[:, 1]

        plt.plot(x, y)
        plt.fill(x, y, alpha=0.5)

    plt.gca().invert_yaxis()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Polygons")

    plt.show()


def binary_mask_to_contours(binary_mask):
    raw_contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = []
    for contour in raw_contours:
        contour = contour.reshape((-1, 2)).tolist()
        contours.append(contour)
    return contours


def polygons_to_mask(polygons, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    for polygon in polygons:
        polygon = np.array([polygon], dtype=np.int32)
        cv2.fillPoly(mask, polygon, 255)
    return mask


def simplify_polygon(polygon, epsilon):
    # Convert the polygon to a NumPy array
    polygon = np.array(polygon, dtype=np.float32)

    # Reshape the polygon array to match the input format of the algorithm
    polygon = polygon.reshape((-1, 1, 2))

    # Apply the Ramer-Douglas-Peucker algorithm to simplify the polygon
    approx_polygon = cv2.approxPolyDP(polygon, epsilon, closed=True)

    # Convert the simplified polygon back to the original format
    approx_polygon = approx_polygon.squeeze().tolist()

    return approx_polygon


def binary_mask_to_polygons(
    mask: np.ndarray, epsilon: float, foreground_value: Any = True
):
    binary_mask = (mask == foreground_value).astype(np.uint8)
    contours = binary_mask_to_contours(binary_mask)
    simplified = []
    for contour in contours:
        simplified.append(simplify_polygon(contour, epsilon))
    return simplified


def mask_to_polygons(mask: np.ndarray, epsilon: float, instance_mask: bool = True):
    if not instance_mask:
        return binary_mask_to_polygons(mask, epsilon)
    polygons = []
    for instance_id in np.unique(mask[mask != 0]):
        polygons.append(binary_mask_to_polygons(mask, epsilon, instance_id))

    return polygons
