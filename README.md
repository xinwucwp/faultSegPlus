# FaultSeg3D plus: A comprehensive study on evaluating and improving CNN-based seismic fault segmentation


**Here we provide both Tensorflow and Pytorch versions of FaultSegPlus (improved from [faultSeg](https://github.com/xinwucwp/faultSeg) by Wu et al. (2019)) for 3D fault segmentation**

As described in **FaultSeg3D plus: A comprehensive study on evaluating and improving CNN-based seismic fault segmentation** 
by [You Li](http://cig.ustc.edu.cn/you/list.htm)<sup>1</sup>, 
[Xinming Wu](http://cig.ustc.edu.cn/xinming/list.htm)<sup>*,1</sup>, 
Zhengyu Zhu,
Jicai Ding, and 
Qingzhen Wang
<sup>1</sup>School of Earth and Space Sciences, USTC; <sup>2</sup>CNOOC.

## Getting Started with Example Model for fault prediction

If you would just like to try out a pretrained example model, then you can download the [pretrained model](https://drive.google.com/file/d/1MjcWmRG6uIZoH4E1_bfa9gXvABwp83kN/view?usp=drive_link) and use the 'apply.py' script to run a demo.


### Dataset

**To train our CNN network, we automatically created 400 pairs of synthetic seismic and corresponding fault volumes, which were shown to be more realistic and diverse than the [200 pairs](https://drive.google.com/open?id=1I-kBAfc_ag68xQsYgAHbqWYdddk4XHHd) published by Wu et al. (2019).** 

**The training and validation datasets will be uploaded soon**

### Training

Run train.py to start training a new faultSeg model by using the 400 synthetic datasets

## Publications

If you find this work helpful in your research, please cite:

    @article{li2025faultSegPlus,
        author = {You Li and Xinming Wu and Zhenyu Zhu and Jicai Ding and Qingzhen Wang},
        title = {Fault{S}eg3{D} plus: A comprehensive study on evaluating and improving CNN-based seismic fault segmentation},
        journal = {GEOPHYSICS},
        volume = {89},
        number = {5},
        pages = {N77–N91},
        year = {2024},
    }

## License

This extension to the Keras library is released under a creative commons license which allows for personal and research use only. 
For a commercial license please contact the authors. You can view a license summary here: http://creativecommons.org/licenses/by-nc/4.0/


