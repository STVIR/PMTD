# PMTD in maskrcnn-benchmark

This project hosts the code for implementing the network structure and plane clustering algorithm of PMTD for text detection, as presented in our paper:
```bibtex
@article{liu2019pyramid,
  title={Pyramid Mask Text Detector},
  author={Liu, Jingchao and Liu, Xuebo and Sheng, Jie and Liang, Ding and Li, Xin and Liu, Qingjie},
  journal={arXiv preprint arXiv:1903.11800},
  year={2019}
}
```
The full paper is available at: [https://arxiv.org/abs/1903.11800](https://arxiv.org/abs/1903.11800). 
## Installation
Check [INSTALL.md](INSTALL.md) for installation instructions.

## Perform testing on ICDAR 2017 MLT dataset

### symlink dataset
We recommend to symlink [ICDAR 2017 MLT](http://rrc.cvc.uab.es/?ch=8) dataset to `datasets/` as follows
```bash
# eg: ~/Projects/PMTD
cd PROJECT_ROOT

mkdir -p datasets/icdar2017mlt
cd datasets/icdar2017mlt

# symlink for images and annotations
ln -s /path_to_icdar2017mlt_dataset/ch8_test_images
```

### generate coco label for dataset
```bash
# ${PWD} = datasets/icdar2017mlt
mkdir annotations
cd PROJECT_ROOT
PYTHONPATH=. python demo/utils/generate_icdar2017.py
# label will output to PROJECT_ROOT/datasets/icdar2017mlt/annotations/test_coco.json
```

### download the pretrained PMTD model
```bash
# ${PWD} = PROJECT_ROOT
mkdir models
wget url_to_model models/PMTD_rectify.pth
```

### test image
In the test stage, we use one GPU of TITANX 11G with a batch size 4. When encountering the out-of-memory (OOM) error, you may need to modify `configs/e2e_PMTD_R_50_FPN_1x_test.yaml` TEST.IMS_PER_BATCH: 4.
```bash
PYTHONPATH=. python tools/test_net.py --config=configs/e2e_PMTD_R_50_FPN_1x_test.yaml
# results will output to PROJECT_ROOT/inference/icdar_2017_mlt_test/
# - bbox.json // when using coco evaluation criterion
# - segm.json // when using coco evaluation criterion
# - dataset.pth
# - predictions.pth
# - results_{scale}.pth, in default setting, scale=1600
```

### convert results to ICDAR 2017 format
```bash
PYTHONPATH=. python demo/utils/convert_results_to_icdar.py
# results will output to PROJECT_ROOT/inference/icdar_2017_mlt_test/
# - icdar.zip
```

### submit icdar.zip to [ICDAR 2017 MLT](http://rrc.cvc.uab.es/?ch=8)

## Perform testing for single image
```bash
cd PROJECT_ROOT
PYTHONPATH=. python demo/PMTD_demo.py --image_path=datasets/icdar2017mlt/ch8_test_images/img_1.jpg
```

## Citations
Please consider citing this project in your publications if it helps your research. The following is a BibTeX reference.
```
@article{liu2019pyramid,
  title={Pyramid Mask Text Detector},
  author={Liu, Jingchao and Liu, Xuebo and Sheng, Jie and Liang, Ding and Li, Xin and Liu, Qingjie},
  journal={arXiv preprint arXiv:1903.11800},
  year={2019}
}
```

## License

PMTD is released under the MIT license. See [LICENSE](LICENSE) for additional details.
