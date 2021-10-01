# Local climate zone classification using a multi-scale, multi-level attention network
[Minho Kim](minho.me), Doyoung Jeong, [Yongil Kim](https://www.researchgate.net/profile/Yongil-Kim-2)
---------------------

The code is provided for the Multi-Scale, Multi-Level Attention Network (MSMLA) proposed in [Local climate zone classification using a multi-scale, multi-level attention network](https://www.sciencedirect.com/science/article/abs/pii/S0924271621002537) accepted in ISPRS J. 

The model is shown below:

![alt text](./images/msmla.jpg)

All experiments were trained from scratch and were performed using an Intel Core i7-6700 CPU at 3.40 GHz and an NVIDIA GeForce RTX 2070 Super Graphics Processor Unit (GPU) with 8 GB of memory. Python 3.7.9 was used with Tensorflow 2.3.0. For training hyperparameters, an early stop of 15 epochs, a learning rate of 0.002, and a decay factor of 0.004 were used. The adaptive moment estimation (adam) optimizer was chosen to minimize the cross-entropy loss function. Filter weights were initialized using “He normal” initialization.

LCZ Classification
---------------------
For comparisons with other **state-of-the-art, deep learning-based LCZ classification models**, refer to the following papers:

1. **CNN** 
Rosentreter, J., Hagensieker, R., & Waske, B. (2020). Towards large-scale mapping of local climate zones using multitemporal Sentinel 2 data and convolutional neural networks. Remote Sensing of Environment, 237, 111472. ([Link])(https://doi.org/10.1016/j.rse.2019.111472)

2. **LCZNet** 
Liu, S., & Shi, Q. (2020). Local climate zone mapping as remote sensing scene classification using deep learning: A case study of metropolitan China. ISPRS Journal of Photogrammetry and Remote Sensing, 164, 229-242. ([Link])(https://doi.org/10.1016/j.isprsjprs.2020.04.008)

3. **Sen2LCZ-Net** 
Qiu, C., Tong, X., Schmitt, M., Bechtel, B., & Zhu, X. X. (2020). Multilevel feature fusion-based CNN for local climate zone classification from sentinel-2 images: Benchmark results on the So2Sat LCZ42 dataset. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 13, 2793-2806. ([Link])(https://ieeexplore.ieee.org/abstract/document/9103196)

Citation
---------------------
**Please cite the journal paper if this code is useful and helpful for your research.**

    @article{kim2021local,
      title={Local climate zone classification using a multi-scale, multi-level attention network},
      author={Kim, Minho and Jeong, Doyoung and Kim, Yongil},
      journal={ISPRS Journal of Photogrammetry and Remote Sensing},
      volume={181C},
      pages={345--366},
      year={2021},
      publisher={Elsevier}
    }
