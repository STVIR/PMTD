from demo.utils.generate_util import GenerateUtil


def generate_cocojson(data_type, image_num: int, output_label_file):
    """

    :param data_type:
    :param image_num:
    :param output_label_file:
    :return:

    Example:
        validate_label_file = path.join(root_dir, f'validate_coco_with_ignore.json')
        generate_cocojson('validation', 1800, validate_label_file)
    """
    generate_util = GenerateUtil(src_info, with_dir_name=False, match_suffix=True, use_ignore=True)
    coco_label = generate_util.get_coco_label()

    insert_annotation = generate_util.insert_factory(data_type)

    for i in range(1, image_num + 1):
        insert_annotation(i, coco_label)

    coco_label.dump(output_label_file)


def generate_cocojson_join(train_image_num, validate_image_num, output_label_file):
    """

    :param train_image_num:
    :param validate_image_num:
    :param output_label_file:
    :return:

    Example:
        label_file = path.join(root_dir, f'train_and_validate_coco_with_ignore.json')
        generate_cocojson_join(train_image_num=7200, validate_image_num=1800, output_label_file=label_file)
    """
    generate_util = GenerateUtil(src_info, with_dir_name=True, match_suffix=True, use_ignore=True)
    coco_label = generate_util.get_coco_label()

    insert_annotation = generate_util.insert_factory('training')
    for i in range(1, train_image_num + 1):
        insert_annotation(i, coco_label)

    insert_annotation = generate_util.insert_factory('validation')
    for i in range(1, validate_image_num + 1):
        insert_annotation(i, coco_label)

    coco_label.dump(output_label_file)


if __name__ == '__main__':
    import os.path as path

    root_dir = 'datasets/icdar2017mlt'
    image_dir_dict = {
        'training': 'ch8_training_images',
        'validation': 'ch8_validation_images',
        'test': 'ch8_test_images'
    }
    image_template_dict = {
        'training': 'img_%d',
        'validation': 'img_%d',
        'test': 'ts_img_%05d'
    }
    label_dir_dict = {
        'training': 'ch8_training_localization_transcription_gt_v2',
        'validation': 'ch8_validation_localization_transcription_gt_v2'
    }
    label_template_dict = {
        'training': 'gt_img_%d.txt',
        'validation': 'gt_img_%d.txt',
    }

    src_info = [root_dir, image_dir_dict, image_template_dict, label_dir_dict, label_template_dict]

    test_label_file = path.join(root_dir, 'annotations', 'test_coco.json')
    generate_cocojson('test', 9000, test_label_file)
