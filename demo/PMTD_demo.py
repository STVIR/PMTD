import argparse
import os

import cv2

from demo.PMTD_predictor import PMTDDemo
from demo.inference import PlaneClustering
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker


def build_parser():
    parser = argparse.ArgumentParser(description="PMTD Single Image Inference")
    parser.add_argument(
        "--image_path",
        default="datasets/icdar2017mlt/ch8_test_images/img_1.jpg",
        metavar="FILE",
        help="path to input image"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="test device"
    )
    parser.add_argument(
        "--longer_size",
        type=int,
        default=1600,
        help="test scale for image"
    )
    parser.add_argument(
        "--method",
        default="PlaneClustering",
        choices=["PlaneClustering", "HardThreshold"],
        help="postprocess method for text mask"
    )
    parser.add_argument(
        "--output_type",
        default="Image",
        choices=["Image", "Points"],
        help="output type for predicted results"
    )
    parser.add_argument(
        "--model_path",
        default="models/PMTD_ICDAR2017MLT.pth",
        metavar="FILE",
        help="path to pretrained model"
    )
    return parser


def create_pmtd_demo(args):
    cfg.merge_from_file("configs/e2e_PMTD_R_50_FPN_1x_ICDAR2017MLT_test.yaml")
    cfg.merge_from_list([
        'MODEL.DEVICE', args.device,
        'MODEL.WEIGHT', args.model_path,
        'INPUT.MAX_SIZE_TEST', args.longer_size,
    ])

    if args.method == 'PlaneClustering':
        masker = PlaneClustering()
    else:
        masker = Masker(threshold=0.01, padding=1)

    pmtd_demo = PMTDDemo(
        cfg,
        masker=masker,
        confidence_threshold=0.5,
        show_mask_heatmaps=False,
    )

    return pmtd_demo


def main():
    parser = build_parser()
    args = parser.parse_args()
    pmtd_demo = create_pmtd_demo(args)
    assert os.path.exists(args.image_path), "No such image"
    image = cv2.imread(args.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if args.output_type == "Image":
        predictions = pmtd_demo.run_on_opencv_image(image)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 800, 800)
        cv2.imshow('image', predictions[:, :, ::-1])
        cv2.waitKey(0)
    else:
        predictions = pmtd_demo.compute_prediction(image)
        top_predictions = pmtd_demo.select_top_predictions(predictions)

        bboxes = top_predictions.bbox
        masks = top_predictions.extra_fields['mask']
        scores = top_predictions.extra_fields['scores']
        for bbox, mask, score in zip(bboxes, masks, scores):
            print(bbox, mask[0], score)


if __name__ == '__main__':
    main()
