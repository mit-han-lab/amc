# AutoML for Model Compression (AMC)

This repo contains the PyTorch implementation for paper [**AMC: AutoML for Model Compression and Acceleration on Mobile Devices**](https://arxiv.org/abs/1802.03494). 

![overview](https://hanlab.mit.edu/projects/amc/images/overview.png)



## Reference

If you find the repo useful, please kindly cite our paper:

```
@inproceedings{he2018amc,
  title={AMC: AutoML for Model Compression and Acceleration on Mobile Devices},
  author={He, Yihui and Lin, Ji and Liu, Zhijian and Wang, Hanrui and Li, Li-Jia and Han, Song},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2018}
}
```

Other papers related to automated model design:

- HAQ: Hardware-Aware Automated Quantization with Mixed Precision ([CVPR 2019](https://arxiv.org/abs/1811.08886))

- ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware ([ICLR 2019](https://arxiv.org/abs/1812.00332))



## Training AMC

Current code base supports the automated pruning of **MobileNet** on **ImageNet**. The pruning of MobileNet consists of 3 steps: **1. strategy search; 2. export the pruned weights; 3. fine-tune from pruned weights**.

To conduct the full pruning procedure, follow the instructions below (results might vary a little from the paper due to different random seed):

1. **Strategy Search**

   To search the strategy on MobileNet ImageNet model, first get the pretrained MobileNet checkpoint on ImageNet by running:

   ```
   bash ./checkpoints/download.sh
   ```

   It will also download our 50% FLOPs compressed model. Then run the following script to search under 50% FLOPs constraint:

   ```bash
   bash ./scripts/search_mobilenet_0.5flops.sh
   ```

   Results may differ due to different random seed. The strategy we found and reported in the paper is:

   ```
   [3, 24, 48, 96, 80, 192, 200, 328, 352, 368, 360, 328, 400, 736, 752]
   ```

2. **Export the Pruned Weights**

   After searching, we need to export the pruned weights by running:

   ```
   bash ./scripts/export_mobilenet_0.5flops.sh
   ```

   Also we need to modify MobileNet file to support the new pruned model (here it is already done in `models/mobilenet.py`)

3. **Fine-tune from Pruned Weights**a

   After exporting, we need to fine-tune from the pruned weights. For example, we can fine-tune using cosine learning rate for 150 epochs by running:

   ```
   bash ./scripts/finetune_mobilenet_0.5flops.sh
   ```



## AMC Compressed Model

We also provide the models and weights compressed by our AMC method. We provide compressed MobileNet-V1 and MobileNet-V2 in both PyTorch and TensorFlow format [here](https://github.com/mit-han-lab/amc-compressed-models). 

Detailed statistics are as follows:

| Models                   | Top1 Acc (%) | Top5 Acc (%) |
| ------------------------ | ------------ | ------------ |
| MobileNetV1-width*0.75   | 68.4         | 88.2         |
| **MobileNetV1-50%FLOPs** | **70.494**   | **89.306**   |
| **MobileNetV1-50%Time**  | **70.200**   | **89.430**   |
| MobileNetV2-width*0.75   | 69.8         | 89.6         |
| **MobileNetV2-70%FLOPs** | **70.854**   | **89.914**   |



## Dependencies

Current code base is tested under following environment:

1. Python 3.7.3
2. PyTorch 1.1.0
3. torchvision 0.2.1
4. NumPy 1.14.3
5. SciPy 1.1.0
6. scikit-learn 0.19.1
7. [tensorboardX](https://github.com/lanpa/tensorboardX)
8. ImageNet dataset



## Contact

To contact the authors:

Ji Lin, jilin@mit.edu

Song Han, songhan@mit.edu
