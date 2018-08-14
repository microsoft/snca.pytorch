## Improving Generalization via Scalable Neighborhood Component Analysis

This repo constains the pytorch implementation for the ECCV2018 paper [(arxiv)](https://arxiv.org/pdf/.pdf).
The project is about deep learning feature representations optimized for
nearest neighbor classifiers, which may generalize to new object categories.

Much of code is borrowed from the previous [unsupervised learning project](https://arxiv.org/pdf/1805.01978.pdf).
Please refer to [this repo](https://github.com/zhirongw/lemniscate.pytorch) for more details.


## Pretrained Model

Currently, we provide 3 pretrained ResNet models.
Each release contains the feature representation of all ImageNet training images (600 mb) and model weights (100-200mb).
You can also get these representations by forwarding the network for the entire ImageNet images.

- [ResNet 18](http://zhirongw.westus2.cloudapp.azure.com/models/snca_resnet18.pth.tar) (top 1 accuracy 70.59%)
- [ResNet 34](http://zhirongw.westus2.cloudapp.azure.com/models/snca_resnet34.pth.tar) (top 1 accuracy 74.41%)
- [ResNet 50](http://zhirongw.westus2.cloudapp.azure.com/models/snca_resnet50.pth.tar) (top 1 accuracy 76.57%)

## Nearest Neighbor

Please follow [this link](http://zhirongw.westus2.cloudapp.azure.com/nn.html) for a list of nearest neighbors on ImageNet.
Results are visualized from our ResNet50 feature, compared with baseline ResNet50 feature, raw image features and supervised features.
First column is the query image, followed by 20 retrievals ranked by the similarity.

## Usage

Our code extends the pytorch implementation of imagenet classification in [official pytorch release](https://github.com/pytorch/examples/tree/master/imagenet). 
Please refer to the official repo for details of data preparation and hardware configurations.

- install python2 and [pytorch=0.3](http://pytorch.org)

- clone this repo: `git clone https://github.com/zhirongw/snca.pytorch`

- Training on ImageNet:

  `python main.py DATAPATH --arch resnet18 -j 32 --temperature 0.05 --low-dim 128 -b 256 `

  - During training, we monitor the supervised validation accuracy by K nearest neighbor with k=1, as it's faster, and gives a good estimation of the feature quality.

- Testing on ImageNet:

  `python main.py DATAPATH --arch resnet18 --resume input_model.pth.tar -e` runs testing with default K=30 neighbors.

- Training on CIFAR10:

  `python cifar.py --nce-t 0.05 --lr 0.1`


## Citation
```
@inproceedings{wu2018improving,
  title={Improving Generalization via Scalable Neighborhood Component Analysis},
  author={Wu, Zhirong and Efros, Alexei A and Yu, Stella},
  booktitle={European Conference on Computer Vision (ECCV) 2018},
  year={2018}
}
```

## Contact

For any questions, please feel free to reach 
```
Zhirong Wu: xavibrowu@gmail.com
```
