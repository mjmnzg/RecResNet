# recresnet
Cross-subject classification of cognitive loads using a recurrent-residual deep network

	Jiménez-Guarneros M., Gómez-Gil P. "Cross-subject classi-
        fication of cognitive loads using a recurrent-residual 
        deep network". IEEE Symposium Series on Computational Inte-
        lligence (IEEE SSCI 2017).

Please consider citing our paper if you find this code useful in your research:

@INPROCEEDINGS{8280897, 
author={M. {Jiménez-Guarneros} and P. {Gómez-Gil}}, 
booktitle={2017 IEEE Symposium Series on Computational Intelligence (SSCI)}, 
title={Cross-subject classification of cognitive loads using a recurrent-residual deep network}, 
year={2017}, 
volume={}, 
number={}, 
pages={1-7}, 
month={Nov}}

https://ieeexplore.ieee.org/document/8280897

Abstract: The problem of automatically learning temporal and spectral feature representations for EEG signals has been intensely studied in the last years. However, most solutions are focused on extracting representations used for training classifiers in particular subjects. This is not well suitable for applications involving several subjects, since it requires high computing times and costs at labeling data and training classifiers for each new subject involved. To address this problem, we propose an improvement of a deep neural network architecture, using residual layers and Gated Recurrent Units (GRU), able to extract feature representations for “cross-subject” classification. Our architecture, called RecResNet, achieved a better accuracy (0.907±0.124) and F-measure (0.896±0.148) than other baseline methods, when applied to the classification of four levels of cognitive loads using 13 subjects.


Packages:

- Python (>= 3.6)
- Tensorflow (>= 1.8)
- NumPy (>= 1.8.2)
- SciPy (>= 0.13.3)

NOTE: Ubuntu 18.10 and Nvidia GTX 1080 were used in our experiments.


Command to execute project:

CUDA_VISIBLE_DEVICES=0 python3 run_main.py --model resrecnet --dataset pbashivan --output outputs/resrecnet
