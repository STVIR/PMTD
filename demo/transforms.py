import cv2
import numpy as np
from PIL import Image


class Resize(object):
    def __init__(self, max_size, with_target=False):
        self.max_size = max_size

        if with_target:
            self.call = self.call_2
        else:
            self.call = self.call_1

    def get_size(self, image_size):
        w, h = image_size
        max_size = self.max_size
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        ratio = max_size / max_original_size
        if w < h:
            ow = int(ratio * min_original_size)
            oh = max_size
        else:
            oh = int(ratio * min_original_size)
            ow = max_size

        return oh, ow

    def resize(self, image, size):
        image = np.array(image)
        image = cv2.resize(image, (size[1], size[0]))
        image = Image.fromarray(image)
        return image

    def call_1(self, image):
        size = self.get_size(image.size)
        image = self.resize(image, size)
        return image

    def call_2(self, image, target):
        size = self.get_size(image.size)
        image = self.resize(image, size)
        target = target.resize(size)
        return image, target

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)
