# We consolidate 11 real labeled datasets and 3 unlabeled datasets.
### [Download preprocessed lmdb dataset for traininig and evaluation](https://www.dropbox.com/sh/1s6r4slurc5ei2n/AACg6TqoDfGdKe8t40Em1fgxa?dl=0)
The detail of datasets is described in [our paper and supplementary materials](https://arxiv.org/abs/2103.04400). <br>
'data_CVPR2021.zip' contains 11 real labeled datasets and 3 unlabeled datasets. <br>
The structure of 'data_CVPR2021' is as follows:
```
data_CVPR2021
├── training
│   ├── label
│   │   ├── real
│   │   │   ├── 1.SVT
│   │   │   ├── 2.IIIT
│   │   │   ├── 3.IC13
│   │   │   ├── 4.IC15
│   │   │   ├── 5.COCO
│   │   │   ├── 6.RCTW17
│   │   │   ├── 7.Uber
│   │   │   ├── 8.ArT
│   │   │   ├── 9.LSVT
│   │   │   ├── 10.MLT19
│   │   │   └── 11.ReCTS
│   │   └── synth (for synthetic data, follow guideline below)
│   │       ├── MJ
│   │       │   ├── MJ_train
│   │       │   ├── MJ_valid
│   │       │   └── MJ_test
│   │       ├── ST
│   │       ├── ST_spe
│   │       └── SA
│   └── unlabel
│       ├── U1.Book32
│       ├── U2.TextVQA
│       └── U3.STVQA
├── validation
│   ├── 1.SVT
│   ├── 2.IIIT
│   ├── 3.IC13
│   ├── 4.IC15
│   ├── 5.COCO
│   ├── 6.RCTW17
│   ├── 7.Uber
│   ├── 8.ArT
│   ├── 9.LSVT
│   ├── 10.MLT19
│   └── 11.ReCTS
└── evaluation
    ├── benchmark
    │   ├── SVT
    │   ├── IIIT5k_3000
    │   ├── IC13_1015
    │   ├── IC15_2077
    │   ├── SVTP
    │   └── CUTE80
    └── addition
        ├── 5.COCO
        ├── 6.RCTW17
        ├── 7.Uber
        ├── 8.ArT
        ├── 9.LSVT
        ├── 10.MLT19
        └── 11.ReCTS
```
- Although we used all of real datasets in our experiments, one may use the part of them instead of all. 
- For synthetic data, [MJSynth(MJ)](http://www.robots.ox.ac.uk/~vgg/data/text/)[1], [SynthText(ST)](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)[2], and ST_spe (ST that contains special characters) are released in the repository [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark). <br>
Download them from [here](https://www.dropbox.com/sh/i39abvnefllx2si/AAAbAYRvxzRp3cIE5HzqUw3ra?dl=0): 'data_lmdb_release.zip' contains MJ and ST. 
Download SynthAdd(SA)[3] from [here](https://www.dropbox.com/s/gugy8xdndwkm2cd/SA.zip?dl=0). <br>
Unzip and move them into above data structure.



## License of datasets
Dataset | License | Message from original authors (for datasets with "No explicit license")
-- | -- | -- 
[SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)[4]     | No explicit license | Not yet
[IIIT](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html)[5]    | No explicit license | Not yet
[IC13](http://rrc.cvc.uab.es/?ch=2)[6]    | No explicit license | IC13 is derivative work from the IC03 and IC05 which do not have explicit license.
[IC15](http://rrc.cvc.uab.es/?ch=4)[7]    | [CC BY 4.0](https://rrc.cvc.uab.es/?ch=4&com=downloads)
[COCO](https://vision.cornell.edu/se3/coco-text-2/)[8]    | [CC BY 4.0](https://vision.cornell.edu/se3/coco-text-2/)
[RCTW](http://rctw.vlrlab.net/dataset/)[9]    | No explicit license | Allowed for both commercial and non-commercial use
[UBer](https://s3-us-west-2.amazonaws.com/uber-common-public/ubertext/index.html)[10]   | [CC BY-SA 4.0](https://s3-us-west-2.amazonaws.com/uber-common-public/ubertext/index.html)
[ArT](https://rrc.cvc.uab.es/?ch=14)[11]    | No explicit license | Non-commercial use only
[LSVT](https://rrc.cvc.uab.es/?ch=16)[12]   | No explicit license | Non-commercial use only
[MLT19](https://rrc.cvc.uab.es/?ch=15)[13]  | [CC BY 4.0](https://rrc.cvc.uab.es/?ch=15&com=downloads)
[ReCTS](https://rrc.cvc.uab.es/?ch=12)[14]  | No explicit license | Not yet
[Book32](https://github.com/uchidalab/book-dataset/)[15] | No explicit license | All book cover images are hosted by and copyright Amazon.com, Inc. The the use of the book cover images is fair use for academic purposes.
[TextVQA](https://textvqa.org/)[16]| [CC BY 4.0](https://textvqa.org/dataset)
[ST-VQA](https://rrc.cvc.uab.es/?ch=11)[17] | [multiple license for multiple subdatasets](https://rrc.cvc.uab.es/?ch=11&com=downloads)

- We would like to redistribute the above datasets (preprocessed version) to facilitate future research.
- Some datasets do not publish explicit licenses. In this case, we ask them for the license via email. That is the 'Message from original authors' (except Book32, the message of Book32 from [here](https://github.com/uchidalab/book-dataset/#disclaimer)).
- We still did not get the information of license for datasets SVT, IIIT, and ReCTS. 
- Please let us know if there is a license issue with dataset redistribution.
In this case, we will remove the dataset and provide the preprocessing code for the dataset.
Some datasets may take some days to be preprocessed, which could be a burden for future research. 
Thus, if possible, we would like to redistribute preprocessed datasets.
- **Please carefully check licenses if you are planning to use them for non-academic or commercial use.**

## Reference
[1] Jaderberg, et al. Synthetic data and artificial neural networks for natural scenetext  recognition. In Workshop on Deep Learning, NIPS, 2014. <br>
[2] Gupta, et al. Synthetic data fortext localisation in natural images. In CVPR, 2016. <br>
[3] Li, et al. Show, Attend and Read: A Simple and Strong Baseline for Irregular Text Recognition. In AAAI, 2019 <br>
[4] Wang, et al. End-to-end scenetext recognition. In ICCV, pages 1457–1464, 2011. <br>
[5] Mishra, et al. Scene text recognition using higher order language priors. In BMVC, 2012. <br>
[6] Karatzas, et al. ICDAR 2013 robust reading competition. In ICDAR, pages 1484–1493, 2013. <br>
[7] Karatzas, et al. ICDAR 2015 competition on ro-bust reading. In ICDAR, pages 1156–1160, 2015. <br>
[8] Veit, et al. Coco-text: Dataset and benchmark for text detection and recognition in natural images. arXiv:1601.07140, 2016. <br>
[9] Shi, et al. Icdar2017 competition on reading chinese text in the wild (rctw-17). In ICDAR, 2017. <br>
[10] Zhang, et al. Uber-text: A large-scale dataset for optical character recognition from street-level imagery. In Scene Understanding Workshop, CVPR, 2017. <br>
[11] Chng, et al. Icdar2019 robust reading challenge on arbitrary-shaped text-rrc-art. In ICDAR, 2019. <br>
[12] Sun, et al. Icdar 2019 competition on large-scale street view text with partial labeling-rrc-lsvt. In ICDAR, 2019. <br> 
[13] Nayef, et al. Icdar2019 robust reading challenge on multi-lingual scene text detection and recognition—rrc-mlt-2019. In ICDAR, 2019. <br>
[14] Zhang, et al. Icdar 2019 robust reading challenge on reading chinese text on signboard. In ICDAR, 2019. <br>
[15] Iwana, et al. Judging a book by its cover. arXiv:1610.09204, 2016. <br>
[16] Singh, et al. Towards vqa models that can read. In CVPR, 2019. <br>
[17] Biten, et al. Scene text visual question answering. In ICCV, 2019.
