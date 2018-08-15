## Improving Generalization via Scalable Neighborhood Component Analysis

This repo constains the pytorch implementation for the ECCV 2018 paper [(paper)](https://arxiv.org/pdf/1808.04699.pdf).
We use deep networks to learn feature representations optimized for nearest neighbor classifiers, which could generalize better for new object categories.
This project is a re-investigation of [Neighborhood Component Analysis (NCA)](http://www.cs.toronto.edu/~fritz/absps/nca.pdf)
with recent technologies to make it scalable to deep networks and large-scale datasets.

Much of code is extended from the previous [unsupervised learning project](https://arxiv.org/pdf/1805.01978.pdf).
Please refer to [this repo](https://github.com/zhirongw/lemniscate.pytorch) for more details.

<img src="http://zhirongw.westus2.cloudapp.azure.com/figs/snca.png" width="800px"/>

## Pretrained Models

Currently, we provide three pretrained ResNet models.
Each release contains the feature representation of all ImageNet training images (600 mb) and model weights (100-200mb).
Models and their performance with nearest neighbor classifiers are as follows.

- [ResNet 18](http://zhirongw.westus2.cloudapp.azure.com/models/snca_resnet18.pth.tar) (top 1 accuracy 70.59%)
- [ResNet 34](http://zhirongw.westus2.cloudapp.azure.com/models/snca_resnet34.pth.tar) (top 1 accuracy 74.41%)
- [ResNet 50](http://zhirongw.westus2.cloudapp.azure.com/models/snca_resnet50.pth.tar) (top 1 accuracy 76.57%)

Code to reproduce the rest of the experiments are comming soon.

## Nearest Neighbors

Please follow [this link](http://zhirongw.westus2.cloudapp.azure.com/nn.html) for a list of nearest neighbors on ImageNet.
Results are visualized from our ResNet50 feature, compared with baseline ResNet50 feature, raw image features and previous unsupervised features.
First column is the query image, followed by 20 retrievals ranked by the similarity.

<img src="http://zhirongw.westus2.cloudapp.azure.com/figs/nn.png" width="800px"/>

## Usage

Our code extends the pytorch implementation of imagenet classification in [official pytorch release](https://github.com/pytorch/examples/tree/master/imagenet). 
Please refer to the official repo for details of data preparation and hardware configurations.

- install python2 and [pytorch>=0.4](http://pytorch.org)

- clone this repo: `git clone https://github.com/Microsoft/snca.pytorch`

- Training on ImageNet:

  `python main.py DATAPATH --arch resnet18 -j 32 --temperature 0.05 --low-dim 128 -b 256 `

  - During training, we monitor the supervised validation accuracy by K nearest neighbor with k=1, as it's faster, and gives a good estimation of the feature quality.

- Testing on ImageNet:

  `python main.py DATAPATH --arch resnet18 --resume input_model.pth.tar -e` runs testing with default K=30 neighbors.

- Memory Consumption and Computation Issues

  Memory consumption is more of an issue than computation time.
  Currently, the implementation of nca module is not paralleled across multiple GPUs.
  Hence, the first GPU will consume much more memory than the others.
  For example, when training a ResNet18 network, GPU 0 will consume 11GB memory, while the others each takes 2.5GB.
  You will need to set the Caffe style "-b 128 --iter-size 2" for training deeper networks.
  Our released models are trained with V100 machines.
  
- Training on CIFAR10:

  `python cifar.py --temperature 0.05 --lr 0.1`


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

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
