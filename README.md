# FaultSeg3D plus: A comprehensive study on evaluating and improving CNN-based seismic fault segmentation


**Here we provide both Tensorflow and Pytorch versions of FaultSegPlus (improved from [faultSeg](https://github.com/xinwucwp/faultSeg) by Wu et al. (2019)) for 3D fault segmentation**

As described in **FaultSeg3D plus: A comprehensive study on evaluating and improving CNN-based seismic fault segmentation** 
by [You Li](http://cig.ustc.edu.cn/you/list.htm)<sup>1</sup>, 
[Xinming Wu](http://cig.ustc.edu.cn/xinming/list.htm)<sup>1</sup>, 
Zhengyu Zhu,
Jicai Ding, and 
Qingzhen Wang
<sup>1</sup>School of Earth and Space Sciences, USTC; <sup>2</sup>CNOOC.

## Getting Started with Example Model for fault prediction

If you would just like to try out a pretrained example model, then you can download the [pretrained model](https://drive.google.com/file/d/1MjcWmRG6uIZoH4E1_bfa9gXvABwp83kN/view?usp=drive_link) and use the 'apply.py' script to run a demo.


### Dataset

**To train our CNN network, we automatically created 400 pairs of synthetic seismic and corresponding fault volumes, which were more realistic and diverse than the [200 pairs](https://drive.google.com/open?id=1I-kBAfc_ag68xQsYgAHbqWYdddk4XHHd) published in 2019.** 

**The training and validation datasets will be uploaded soon**

### Training

Run train.py to start training a new faultSeg model by using the 200 synthetic datasets

## Publications

If you find this work helpful in your research, please cite:

    @article{wu2019faultSeg,
        author = {Xinming Wu and Luming Liang and Yunzhi Shi and Sergey Fomel},
        title = {Fault{S}eg3{D}: using synthetic datasets to train an end-to-end convolutional neural network for 3{D} seismic fault segmentation},
        journal = {GEOPHYSICS},
        volume = {84},
        number = {3},
        pages = {IM35-IM45},
        year = {2019},
    }

## License

This extension to the Keras library is released under a creative commons license which allows for personal and research use only. 
For a commercial license please contact the authors. You can view a license summary here: http://creativecommons.org/licenses/by-nc/4.0/


