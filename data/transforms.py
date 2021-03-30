from .utils import *


class Transforms:
    def __init__(self, min_size, max_size, is_training):
        self.min_size = min_size
        self.max_size = max_size
        self.is_training = is_training

    def __call__(self, img, bbox, label):
        original_size = img.shape[1:]

        # resize
        img = resize_image(img, self.min_size, self.max_size)
        transformed_size = img.shape[1:]

        # normalize
        img = normalize_image(img)

        # calc scale
        scale = transformed_size[0] / original_size[0]

        if self.is_training:
            # resize bbox
            bbox = resize_bbox(bbox, original_size, transformed_size)

            # horizontal flip
            img, flip = horizontal_flip_image(img)
            bbox = horizontal_flip_bbox(bbox, transformed_size, flip)

        return img, bbox, label, scale
