# Lego_segmentation
Semantic segmentation pipeline for Lego blocks (DMIA DL Fall 2019) - 1 class and background
https://www.kaggle.com/c/dmia-dl-aut19-segmentation/overview

Using:

PyTorch Lightning framework

PSPNet with ResNext101-32x8d (pretrained on the imagenet) as a backbone (segmentation_model_pytorch)

Metric - IoU, loss - soft dice
