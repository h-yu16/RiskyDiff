# RiskyDiff

Official repo for AAAI 2026 paper "Generating Risky Samples with Conformity Constraints via Diffusion Models". 

This repo is built upon StableDiffusion, and the environment and checkpoints can also be referred to in StableDiffusion. The main modifications we made are in `riskydiff.py` and `ddim.py`. 

We provide a running script for ImageNet. For other datasets, you may modify the code and files according to your need. 

## Usage

Before you start, you need to download the validation dataset of ImageNet. Then you need to put txt files that store the paths of images of each category in a directory specified by the argument `--txtdir` of `extract.py`. An example of txt files of categories in ImageNet can be downloaded [here](https://cloud.tsinghua.edu.cn/f/fc2d3bf84d7a441eb251/). You can also modify the prefix of paths in txt files to the exact directory of ImageNet data. Besides, you need to download the file of category names of ImageNet also via the previous link and set the path of it via `IMAGENET_LABELS_FILE` in `datastuff.py`. 

For a specific target model, e.g. EfficientNet-B2, you need to first extract CLIP features and other information for the target model. 

```shell
CUDA_VISIBLE_DEVICES=0 python extract.py  --dataset ImageNet --model_arch effnetb2
```

Then you can generate risky samples for the target model. 

```shell
CUDA_VISIBLE_DEVICES=0 python scripts/riskydiff.py --dataset ImageNet --category gorilla --prompt gorilla --gradient_scale 10 --risky_model_arch effnetb2 --match_scale 0.0001 --save_model_eval
```

Note that by setting the argument `--save_model_eval`, the error predictor will be trained and saved. If an error predictor has already been saved, you can replace `--save_model_eval` with `--load_model_eval` to avoid repeated training of the error predictor. 

After generating risky samples, you can get the predictions of the target model for these samples.

```shell
CUDA_VISIBLE_DEVICES=0 python test.py --dataset ImageNet --datadir outputs/samples --category gorilla --model_arch effnetb2
```



