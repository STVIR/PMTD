import collections
import os
import zipfile
from typing import List, Dict, DefaultDict
from zipfile import ZipFile

import cv2
import numpy as np
import pyclipper
import torch


def get_results(path_list, merge_path, image_num, regenerate=True):
    if regenerate or not os.path.exists(merge_path):
        results_merge: DefaultDict[int, list] = collections.defaultdict(list)
        for result_path in path_list:
            result = torch.load(result_path)
            for i in range(1, image_num + 1):
                results_merge[i].append(result[i])
        torch.save(results_merge, merge_path)

    results_merge = torch.load(merge_path)
    return results_merge


def nms(items: List, threshold: float):
    def inter_of_union(item_a: Dict, item_b: Dict):
        bbox_a = item_a['points']
        bbox_b = item_b['points']
        try:
            pc = pyclipper.Pyclipper()
            pc.AddPath(bbox_a, pyclipper.PT_SUBJECT, True)
            pc.AddPath(bbox_b, pyclipper.PT_CLIP, True)
            solution = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
            if len(solution) == 0:
                return 0
            solution = np.asarray(solution[0])
            inter_area = cv2.contourArea(solution)
            a_area = cv2.contourArea(bbox_a)
            b_area = cv2.contourArea(bbox_b)
            union_area = a_area + b_area - inter_area
            try:
                return inter_area / union_area
            except:
                return 1
        except Exception as e:
            print(f"Error: {str(e)}, bbox_a: {bbox_a}, bbox_b: {bbox_b}")
            exit(0)

    length = len(items)
    keep = np.ones(length, dtype=bool)
    for i in range(length):
        if not keep[i]:
            continue
        for j in range(i + 1, length):
            if not keep[j]:
                continue
            if inter_of_union(items[i], items[j]) > threshold:
                keep[j] = False
    return np.asarray(items)[keep]


def output_results(output_dir, scales, image_num, cls_threshold_list, nms_threshold, line_format, pred_file_template):
    os.chdir(output_dir)
    result_merge_path = 'results_multi_scale.pth'
    result_path_list = [f'results_{tag}.pth' for tag in scales]
    results: dict = get_results(result_path_list, result_merge_path, image_num, regenerate=True)
    with ZipFile('icdar.zip', 'w', compression=zipfile.ZIP_DEFLATED) as result_zip:
        for image_id, item_lists in results.items():
            items = []
            for item_list, cls_threshold in zip(item_lists, cls_threshold_list):
                # item {cls_score: float, points: [4, 2]}
                items.extend([item for item in item_list if item['cls_score'] > cls_threshold])
            items = sorted(items, key=lambda x: x['cls_score'], reverse=True)
            items = nms(items, nms_threshold)
            lines = []
            for item in items:
                bbox = item['points'].ravel().astype(np.str).tolist()
                score = item['cls_score']
                line = line_format(bbox, score)
                lines.append(line)
            result_zip.writestr(pred_file_template % image_id, str.join('', lines))


if __name__ == '__main__':
    # postprocess_config = {
    #     'output_dir': None,
    #     'scales': [],
    #     'image_num': 0,
    #     'cls_threshold_list': [],
    #     'nms_threshold': 0,
    #     'line_format': lambda bbox, cls_score: None,
    #     'pred_file_template': None
    # }
    postprocess_config_17 = {
        'output_dir': 'inference/icdar_2017_mlt_test',
        'scales': [1600],
        'image_num': 9000,
        'cls_threshold_list': [0.5],
        'nms_threshold': 0.1,
        'line_format': lambda bbox, cls_score: f"{','.join(bbox)}, {cls_score}\n",
        'pred_file_template': 'res_img_%05d.txt'
    }
    postprocess_config_15 = {
        'output_dir': 'inference/icdar_2015_test',
        'scales': [1920],
        'image_num': 500,
        'cls_threshold_list': [0.4],
        'nms_threshold': 0.3,
        'line_format': lambda bbox, cls_score: f"{','.join(bbox)}\n",
        'pred_file_template': 'res_img_%d.txt'
    }
    output_results(**postprocess_config_17)
