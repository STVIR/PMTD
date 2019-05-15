import cv2
import numpy as np
from torchvision import transforms as T

from demo.inference import PlaneClustering
from demo.predictor import COCODemo
from demo.transforms import Resize
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.utils import cv2_util


class PMTDDemo(COCODemo):
    CATEGORIES = [
        "__background",
        "text"
    ]

    def __init__(self, cfg, masker, **kwargs):
        assert isinstance(masker, Masker)
        super().__init__(cfg, masker, **kwargs)

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg
        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )
        transform = T.Compose(
            [
                T.ToPILImage(),
                Resize(max_size=cfg.INPUT.MAX_SIZE_TEST),
                T.ToTensor(),
                normalize_transform,
            ]
        )
        return transform

    def overlay_mask(self, image, predictions):
        """
        Adds the instances contours for each predicted object.
        Each label has a different color.

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask` and `labels`.
        """
        masks = predictions.get_field("mask").numpy()
        labels = predictions.get_field("labels")

        colors = self.compute_colors_for_labels(labels).tolist()

        if isinstance(self.masker, PlaneClustering):
            for mask, color in zip(masks, colors):
                contours = [mask.reshape(-1, 1, 2).astype(np.int32)]
                image = cv2.drawContours(image, contours, -1, color, 3)
        else:
            for mask, color in zip(masks, colors):
                thresh = mask[0, :, :, None]
                contours, hierarchy = cv2_util.findContours(
                    thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                image = cv2.drawContours(image, contours, -1, color, 3)
        return image
