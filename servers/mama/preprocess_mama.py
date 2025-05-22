from typing import Iterable
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import einops
from mama.preprocess import keep_only_breast
import torch


def otsu_mask(img):
    median = np.median(img)
    _, thresh = cv2.threshold(img, median, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


class OtsuCut(object):

    def __init__(self):
        super().__init__()

    def __process__(self, x):
        if isinstance(x, Image.Image):
            x = np.array(x)
        mask = otsu_mask(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY))
        # Convert to NumPy array if not already

        # Check if the matrix is empty or has no '1's
        if mask.size == 0 or not np.any(mask):
            return Image.fromarray(x)

        # Find the rows and columns where '1' appears
        rows = np.any(mask == 255, axis=1)
        cols = np.any(mask == 255, axis=0)

        # Find the indices of the rows and columns
        min_row, max_row = np.where(rows)[0][[0, -1]]
        min_col, max_col = np.where(cols)[0][[0, -1]]

        # Crop and return the submatrix
        x = x[min_row : max_row + 1, min_col : max_col + 1]
        img = Image.fromarray(x)
        return img

    def __call__(self, x):
        if isinstance(x, Iterable):
            return [self.__process__(im) for im in x]
        else:
            return self.__process__(x)

class KeepOnlyBreast(object):
    def __init__(self):
        super().__init__()


    def __process__(self, x):
        if isinstance(x, Image.Image):
            x = np.array(x)

        x, _ = keep_only_breast(x)
        x = einops.repeat(x, 'h w -> h w 3')

        img = Image.fromarray(x)
        return img

    def __call__(self, x):
        return self.__process__(x)

class Pad(object):
    def __init__(self):
        super().__init__()

    def pad(self, image, ar=1):
        n_rows, n_cols = image.shape[:2]
        image_ratio = n_rows / n_cols
        if image_ratio == ar:
            return image
        if ar < image_ratio:
            new_n_cols = int(n_rows / ar)
            ret_val = np.zeros((n_rows, new_n_cols, image.shape[2]), dtype=image.dtype)
        else:
            new_n_rows = int(n_cols * ar)
            ret_val = np.zeros((new_n_rows, n_cols, image.shape[2]), dtype=image.dtype)
        ret_val[:n_rows, :n_cols] = image
        return ret_val


    def __process__(self, x):
        if isinstance(x, Image.Image):
            x = np.array(x)

        x = self.pad(x)

        img = Image.fromarray(x)
        return img

    def __call__(self, x):
        if isinstance(x, Iterable):
            return [self.__process__(im) for im in x]
        else:
            return self.__process__(x)

class MamaTransform(object):
    def __init__(
        self, is_train: bool = True, img_size: int = 518
    ):
        self.data_transforms = transforms.Compose(
            [
                KeepOnlyBreast(),
                OtsuCut(),
                Pad(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __call__(self, img):
        return self.data_transforms(img)

