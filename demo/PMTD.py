import cv2
import numpy as np
from torchvision import transforms as T

from demo.inference import PlaneClustering
from demo.predictor import COCODemo
from demo.transforms import Resize
from maskrcnn_benchmark.config import cfg
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

        composite = image

        return composite


temp_config = [
    'MODEL.DEVICE', 'cuda',
    'INPUT.MAX_SIZE_TEST', 640,
]

if __name__ == '__main__':
    # update the config options with the config file
    cfg.merge_from_file("configs/e2e_PMTD_R_50_FPN_1x_test.yaml")
    # manual override some options
    cfg.merge_from_list(temp_config)

    image = cv2.imread('datasets/icdar2017mlt/ch8_validation_images/img_1.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # masker = Masker(threshold=0.01, padding=1)
    masker = PlaneClustering()

    pmtd_demo = PMTDDemo(
        cfg,
        masker=masker,
        confidence_threshold=0.5,
        show_mask_heatmaps=False,
    )
    show = True
    if show:
        # load image and then run prediction
        predictions = pmtd_demo.run_on_opencv_image(image)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 800, 800)
        cv2.imshow('image', predictions[:, :, ::-1])
        cv2.waitKey(0)
    else:
        predictions = pmtd_demo.compute_prediction(image)
        top_predictions = pmtd_demo.select_top_predictions(predictions)
        for bbox, score in zip(top_predictions.bbox, top_predictions.extra_fields['scores']):
            print(bbox, score)
