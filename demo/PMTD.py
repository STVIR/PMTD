import cv2
from torchvision import transforms as T

from demo.predictor import COCODemo
from demo.transforms import Resize
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker


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
    pmtd_demo = PMTDDemo(
        cfg,
        masker=Masker(threshold=0.01, padding=1),
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
