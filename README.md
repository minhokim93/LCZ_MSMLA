# Local climate zone classification using a multi-scale, multi-level attention network
[Minho Kim](minho.me), Doyoung Jeong, [Yongil Kim](https://www.researchgate.net/profile/Yongil-Kim-2)
---------------------

The code is provided for the Multi-Scale, Multi-Level Attention Network (MSMLA) proposed in [Local climate zone classification using a multi-scale, multi-level attention network](https://www.sciencedirect.com/science/article/abs/pii/S0924271621002537) accepted in ISPRS J. 

The model is shown below:

![alt text](./images/msmla.jpg)

All experiments were trained from scratch and were performed using an Intel Core i7-6700 CPU at 3.40 GHz and an NVIDIA GeForce RTX 2070 Super Graphics Processor Unit (GPU) with 8 GB of memory. Python 3.7.9 was used with Tensorflow 2.3.0. For training hyperparameters, an early stop of 15 epochs, a learning rate of 0.002, and a decay factor of 0.004 were used. The adaptive moment estimation (adam) optimizer was chosen to minimize the cross-entropy loss function. Filter weights were initialized using “He normal” initialization.

LCZ Classification
---------------------

| Model | Trainable Parameters | OA (%) | WA (%) | OA<sub>BU</sub> (%) | OA<sub>N</sub> (%) | F1 (%) |
|      :---:       |      :---:       |      :---:       |      :---:       |      :---:       |      :---:       |      :---:       |
| [Sen2LCZ](https://ieeexplore.ieee.org/abstract/document/9103196) | 793,348	| 81.0 | 73.9 | 54.4 | 84.8 | 78.7 |
| [LCZNet](https://doi.org/10.1016/j.isprsjprs.2020.04.008) | 3,181,809 | 83.0 | 79.8	| 63.8 | 86.5 | 82.2 | 
| [CNN](https://doi.org/10.1016/j.rse.2019.111472) |   235,105 | 83.4 | 78.1 | 61.1 | 89.0 | 82.2 |
| MSMLA-18<sup>*</sup> | 181,867 | 84.4 | 80.4 | 64.7 | 89.4 | 83.5 |
| <b>MSMLA-50<sup>*</sup></b> | <b>808,913</b> | <b>87.1</b> | <b>85.0</b> | <b>72.4</b> | <b>91.8</b> | <b>86.5</b> |

<sup>*</sup> Proposed models


---------------------
**:fire: State-of-the-art, Deep Learning-based LCZ Classification Models**
1. **Sen2LCZ-Net** 
Qiu, C., Tong, X., Schmitt, M., Bechtel, B., & Zhu, X. X. (2020). Multilevel feature fusion-based CNN for local climate zone classification from sentinel-2 images: Benchmark results on the So2Sat LCZ42 dataset. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 13, 2793-2806.
2. **LCZNet** 
Liu, S., & Shi, Q. (2020). Local climate zone mapping as remote sensing scene classification using deep learning: A case study of metropolitan China. ISPRS Journal of Photogrammetry and Remote Sensing, 164, 229-242. 
3. **CNN** 
Rosentreter, J., Hagensieker, R., & Waske, B. (2020). Towards large-scale mapping of local climate zones using multitemporal Sentinel 2 data and convolutional neural networks. Remote Sensing of Environment, 237, 111472.

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
