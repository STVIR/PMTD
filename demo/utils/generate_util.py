import functools
import json
import os.path as path

import cv2
import numpy as np


def exist_file_path(file_id):
    if not path.exists(file_id):
        raise FileNotFoundError(f'index: {file_id}\n')
    else:
        image_path = path.abspath(file_id)
        return image_path


def match_file_path(file_id):
    file_type = None

    if path.exists(file_id + '.jpg'):
        file_type = '.jpg'
    elif path.exists(file_id + '.JPG'):
        file_type = '.JPG'
    elif path.exists(file_id + '.png'):
        file_type = '.png'
    elif path.exists(file_id + '.PNG'):
        file_type = '.PNG'
    elif path.exists(file_id + '.gif'):
        pass
    elif path.exists(file_id + '.GIF'):
        pass
    else:
        raise FileNotFoundError(f'index: {file_id}\n')

    if file_type is None:
        image_path = None
    else:
        image_path = path.abspath(file_id + file_type)
    return image_path


class CocoLabel:
    def __init__(self, item_info, item_licenses, item_categories):
        self.item_info = item_info
        self.item_licenses = item_licenses
        self.item_categories = item_categories
        self.items_image = []
        self.items_annotation = []

    def dump(self, dst_label_file):
        with open(dst_label_file, 'w') as f:
            json.dump({'info': self.item_info, 'licenses': self.item_licenses, 'categories': self.item_categories,
                       'images': self.items_image, 'annotations': self.items_annotation}, f)


class GenerateUtil:
    def __init__(self, src_info, with_dir_name, match_suffix, use_ignore):
        self.src_info = src_info
        self.with_dir_name = with_dir_name
        self.match_suffix = match_suffix
        self.use_ignore = use_ignore

    @staticmethod
    def generate_item_info():
        info = {
            "description": "ICDAR 2017 MLT Dataset",
            "url": "http://rrc.cvc.uab.es",
            "version": "0.1.0",
            "year": 2018,
            "contributor": "Jingchao Liu",
            "date_created": "2018/11/30"
        }
        return info

    @staticmethod
    def generate_item_licenses():
        licenses = [
            {
                "id": 1,
                "name": "ICDAR 2017 MLT",
                "url": "http://rrc.cvc.uab.es"
            }
        ]
        return licenses

    @staticmethod
    def generate_item_categories():
        categories = [
            {
                'id': 1,
                'name': 'text',
                'supercategory': 'instance',
            }
        ]
        return categories

    @staticmethod
    def generate_item_image(items_image, file_name, image_size):
        image_info = {
            "id": len(items_image) + 1,
            "file_name": file_name,
            "width": image_size[1],
            "height": image_size[0],
            "date_captured": '2018-11-30 16:00:00',
            "license": 1,
            "coco_url": '',
            "flickr_url": ''
        }
        items_image.append(image_info)

    @staticmethod
    def generate_item_fake_annotation(items_annotation, image_id, i, image_size=None):
        segmentation = [[1, 1, 1, 1, 1, 1, 1, 1]]
        bounding_box = [0, 0, 1, 1]
        annotation_info = {
            "id": image_id,
            "image_id": image_id,
            "category_id": 1,
            "iscrowd": 0,
            "area": 1,
            "bbox": bounding_box,
            "segmentation": segmentation,
            "width": 1,
            "height": 1,
        }
        return items_annotation.append(annotation_info)

    @staticmethod
    def generate_item_true_annotation(items_annotation, image_id, image_index, image_size, label_path_template, use_ignore):
        label_path = path.abspath(label_path_template % image_index)
        with open(label_path) as f:
            for line in f.readlines():
                data = line.split(',')
                segmentation = np.asarray(data[:8], dtype=int)
                iscrowd = 0 if data[9] != '###\n' else 1
                points = segmentation.reshape((-1, 2))
                segmentation = [segmentation.tolist()]
                area = cv2.contourArea(points)
                bounding_box = cv2.boundingRect(points)  # [x, y, w, h]
                annotation_info = {
                    "id": len(items_annotation) + 1,
                    "image_id": image_id,
                    "category_id": 1,
                    "iscrowd": 0 if not use_ignore else iscrowd,
                    "area": area,
                    "bbox": bounding_box,
                    "segmentation": segmentation,
                    "width": image_size[1],
                    "height": image_size[0],
                }
                items_annotation.append(annotation_info)

    def get_coco_label(self):
        item_info = self.generate_item_info()
        item_licenses = self.generate_item_licenses()
        item_categories = self.generate_item_categories()
        coco_label = CocoLabel(item_info, item_licenses, item_categories)
        return coco_label

    def insert_factory(self, data_type: str):
        def _insert_annotation(i, coco_label: CocoLabel):
            image_path = get_file_path(file_id=image_path_template % i)
            # skip gif file format
            if image_path is None:
                return
            image_name = get_image_name(image_path)
            image_size = np.shape(cv2.imread(image_path))[:2]
            generate_item_image(coco_label.items_image, image_name, image_size)
            generate_item_annotation(coco_label.items_annotation, len(coco_label.items_image), i, image_size)

        root_dir, image_dir_dict, image_template_dict, label_dir_dict, label_template_dict = self.src_info
        assert data_type in image_dir_dict.keys()

        image_path_template = path.join(root_dir, image_dir_dict[data_type], image_template_dict[data_type])

        generate_item_image = self.generate_item_image

        if data_type in label_dir_dict:
            label_path_template = path.join(root_dir, label_dir_dict[data_type], label_template_dict[data_type])
            generate_item_annotation = functools.partial(self.generate_item_true_annotation,
                                                         label_path_template=label_path_template,
                                                         use_ignore=self.use_ignore)
        else:
            generate_item_annotation = self.generate_item_fake_annotation

        if self.with_dir_name:
            get_image_name = lambda image_path: path.join(image_dir_dict[data_type], path.basename(image_path))
        else:
            get_image_name = lambda image_path: path.basename(image_path)

        if self.match_suffix:
            get_file_path = match_file_path
        else:
            get_file_path = exist_file_path

        return _insert_annotation
