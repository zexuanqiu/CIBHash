# CIBHash

A Pytorch implementation of paper "**Unsupervised Hashing with Contrastive Information Bottleneck** "

### Main Dependencies

- torch 1.4.0
- torchvision 0.5.0
- Pillow 5.4.1
- opencv-python 4.5.1.48



### How to Run

```shell
# Run with the Cifar10 dataset
python main.py cifar10 --train --dataset cifar10 --encode_length 16 --cuda
```

If you run the above command, the program will download the Cifar10 dataset to the directory `./data/cifar10/` and then start to train. 

Moreover, you can find the download link of  NUS-WIDE dataset in [this page](https://github.com/jiangqy/ADSH-AAAI2018/tree/master/ADSH_matlab); as for the MSCOCO dataset, you can directly visit the [homepage](https://cocodataset.org/#download) to get the source data.  You can refer to `./utils/data.py` to get hints of preprocessing these two datasets.

