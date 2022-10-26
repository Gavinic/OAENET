# <center> OAENet: Oriented Attention Ensemble for Accurate Facial Expression Recognition

<h4 align="center">Zhengning Wang, Fanwei Zeng, Shuaicheng Liu, Bing Zeng</h4>
<h4 align="center">University of Electronic Science and Technology of China</h4>

This is the official implementation of paper "[***OAENet: Oriented Attention Ensemble for Accurate Facial Expression Recognition***](https://www.sciencedirect.com/science/article/abs/pii/S0031320320304970)" in Pattern Recognition 2021.

## Abstract
Facial Expression Recognition (FER) is a challenging yet important research topic owing to its significance with respect to its academic and commercial potentials. In this work, we propose an oriented attention pseudo-siamese network that takes advantage of global and local facial information for high accurate FER. Our network consists of two branches, a maintenance branch that consisted of several convolutional blocks to take advantage of high-level semantic features, and an attention branch that possesses a UNet- like architecture to obtain local highlight information. Specifically, we first input the face image into the maintenance branch. For the attention branch, we calculate the correlation coefficient between a face and its sub-regions. Next, we construct a weighted mask by correlating the facial landmarks and the corre- lation coefficients. Then, the weighted mask is sent to the attention branch. Finally, the two branches are fused to output the classification results. As such, a direction-dependent attention mechanism is es- tablished to remedy the limitation of insufficient utilization of local information. With the help of our attention mechanism, our network not only grabs a global picture but can also concentrate on important local areas. Experiments are carried out on 4 leading facial expression datasets. Our method has achieved a very appealing performance compared to other state-of-the-art methods.


## Pipeline
![pipeline](https://user-images.githubusercontent.com/1344482/181180943-f48794d7-c499-4919-8f8a-e53eecf3659d.JPG)
Our network pipeline. (a) input image. (b) weighted mask calculated from oriented gradient. Our network consists of a (c) maintenance branch and (d) attention branch, in which (c) consists of several inverted residual and linear bottleneck structures, and (d) is an UNet-like architecture. The maintenance branch takes the original face image as input, and then convolve it to 56 ×56 ×32. The attention branch takes the weighted mask as input. Through downsampling and upsampling, the size of final feature map is also 56 ×56 ×32. The feature map from two branches is fused by depthwise product. (e) is the stackblock that consists of several CNN blocks to produce the final score (f)


## Citations
If you think this work is helpful, please cite

```
@article{wang2021oaenet,
  title={OAENet: Oriented attention ensemble for accurate facial expression recognition},
  author={Wang, Zhengning and Zeng, Fanwei and Liu, Shuaicheng and Zeng, Bing},
  journal={Pattern Recognition},
  volume={112},
  pages={107694},
  year={2021},
  publisher={Elsevier}
}
```

## Installation
* Download the source code by
```
git clone https://github.com/Gavinic/OAENET.git
cd OAENET
```

* Install the package
```
conda create -n OAENET python=3.8
conda activate OAENET
pip install -r requirements.txt
```
## Training
  * datasets 
  You need to download the facial expression dataset AffectNet,Ck+, RAF-DB from the public official website. then the images for traing should be organized   in the following format:
  ```
  xxx.jpg\tlabel
  ```
  label: 0, 1, 2, 3, 4, 5, 6, 7; which represent diffenent facial expression: 'surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral'.
  * strat training:
  ```
  python train.py
  ```
 
 ## Inference
  * For inference, you need to modify the parameter with your own image/model path：
   ```
  self.data_dir = "xxx"  # image floder
  self.model_path = 'xxx'  # model path   xxx.pth

  ```
  Then you can use the following command for inference:
   ```
  python test.py
  ```
