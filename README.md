# Self-Validation

## Self-Validation: Early Stopping for Single-Instance Deep Generative Priors

This is the official implementation of our paper *Self-Validation: Early Stopping for Single-Instance Deep Generative Priors* which has been accepted by the [32nd British Machine Vision Conference 2021](http://www.bmvc2021.com/). You can find our paper via [this link](https://arxiv.org/abs/2110.12271) and the project webpage via [this link](https://sun-umn.github.io/Self-Validation/).


## Set Up the Environment

1. Get and clone the github repository:

   `git clone https://github.com/sun-umn/Self-Validation`

2. Switch to `Self-Validation` :

   `cd XXX/Self-Validation`  
   (*Note*: `XXX` here indicates the upper directory of `Self-Validation`. For example, if you clone `Self-Validation` under `/home/Download`, then you should replace `XXX` with `/home/Download`.)

3. Deactivate conda base environment first you are in (otherwise, go to step 4 directly) (We use [Anaconda3](https://www.anaconda.com/products/individual-d)):

   `conda deactivate`

4. Create a new conda environment with the YML file we provide:

    `conda env create -f environment.yml`
   
5.  Activate conda environment and you are now ready to explpre the codes/models!
    
    `conda activate pytorch_py3.6`
    
## Explore the Codes/Models

### Dataset
We provide a ready-to-use dataset under the folder `/Dataset` where there are 4 types of noises we used in our paper where "XXX_2" indicates "low noise level", "XXX_3" indicates "medium noise level", and "XXX_4" indicates "high noise level".

Alternatively, you can also create the dataset by yourself. The clean images can be downloaded [here](https://webpages.tuni.fi/foi/GCF-BM3D/index.html#ref_results). After you have the clean images, you can follow the [ImageNet-C protocol](https://github.com/hendrycks/robustness) (or write corruption functions by yourself) to create the corrupted images. For the parameters for each noise level, please check the Appendix of [our paper](https://arxiv.org/abs/2110.12271).

### DIP+AE

`DIP+AE` uses [deep image prior (DIP)](https://ieeexplore.ieee.org/abstract/document/8579082) to reconstruct image while simultaneously we train a deep autoencoder to monitor the quality of the reconstructed images and signal early-stopping when DIP starts to recover noise. Inside it, we provide the complete code for image denoising on each noise type and each noise level where they share the same file structures and the merely difference is that each process different noises (or noise levels). Thus, we only use Gaussian noise level 2 (low noise level) as an example below.

**The structure of files**:
```
.
├── Gaussian_2/w256_P500                  /* The image denoising code for Guassian noise level 2 (low noise level).
|   ├── models                            /* The DIP models (adopted direclty from DIP).
|   ├── utils                             /* The DIP functions/packages (adopted direclty from DIP).
|   ├── Main_Start.py                     /* The start point of our method (everhtying starts from this python file).
|   ├── train_denoising.py                /* Train DIP to do the denoising (it will also call our detection AE).
|   ├── util.py                           /* The auxiliary functions for DIP (e.g., preparing the dataset).
|   ├── AE_train.py                       /* The entry of our detection AE; it is called by "train_denoising.py".
|   ├── AE_bp.py                          /* Train our detection AE.
|   ├── AE_model.py                       /* Our detection AE model.
|   ├── AE_util.py                        /* The auxiliary functions for detection AE (e.g., preparing the dataset).
|   ├── Early_Stop.py                     /* Our early stopping function.
|   ├── track_rank.py                     /* Check the rank of the latent code of our detection AE.
|   ├── Smooth_Values.py                  /* To get a smooth curve of AE reconstruction error (we do not use it in our experiments).
|   └── psnr_ssim.py                      /* To get the PSNR and SSIM values.    
└── 
```

We have provided the detailed inline comments in each Python file. You can modify any parameters you want to explore the models. Or, you can try our early-stopping method by simply running the command below:

```
python Main_Start.py
```

### DD+AE

`DD+AE` uses [deep decoder (DD)](https://openreview.net/forum?id=rylV-2C9KQ) to reconstruct image while simultaneously we train a deep autoencoder to monitor the quality of the reconstructed images and signal early-stopping when DD starts to recover noise. Inside it, we provide the complete code for image denoising on each noise type and each noise level where they share the same file structures and the merely difference is that each process different noises (or noise levels).

Since we follow the exact same protocol as that of `DIP+AE` to orgianize files and set up our code for `DD+AE`, we therefore omit the detailed description in here. Please refer [`DIP+AE`](https://github.com/sun-umn/Self-Validation/blob/main/README.md#dipae) for the detailed description.

## Citation/BibTex

More technical details and experimental results can be found in our paper:

Taihui Li, Zhong Zhuang, Hengyue Liang, Le Peng, Hengkang Wang, Ju Sun. Self-Validation: Early Stopping for Single-Instance Deep Generative Priors. 32nd British Machine Vision Conference 2021.

```
@inproceedings{DBLP:conf/bmvc/LiZLPWS21,
  author       = {Taihui Li and
                  Zhong Zhuang and
                  Hengyue Liang and
                  Le Peng and
                  Hengkang Wang and
                  Ju Sun},
  title        = {Self-Validation: Early Stopping for Single-Instance Deep Generative
                  Priors},
  booktitle    = {32nd British Machine Vision Conference 2021, {BMVC} 2021, Online,
                  November 22-25, 2021},
  pages        = {108},
  publisher    = {{BMVA} Press},
  year         = {2021},
  url          = {https://www.bmvc2021-virtualconference.com/assets/papers/1633.pdf},
  timestamp    = {Wed, 22 Jun 2022 16:52:45 +0200},
  biburl       = {https://dblp.org/rec/conf/bmvc/LiZLPWS21.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

```
@article{DBLP:journals/corr/abs-2112-06074,
  author       = {Hengkang Wang and
                  Taihui Li and
                  Zhong Zhuang and
                  Tiancong Chen and
                  Hengyue Liang and
                  Ju Sun},
  title        = {Early Stopping for Deep Image Prior},
  journal      = {CoRR},
  volume       = {abs/2112.06074},
  year         = {2021},
  url          = {https://arxiv.org/abs/2112.06074},
  eprinttype    = {arXiv},
  eprint       = {2112.06074},
  timestamp    = {Mon, 03 Jan 2022 15:45:35 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2112-06074.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

## Contact
- Taihui Li, lixx5027@umn.edu, [https://taihui.github.io/](https://taihui.github.io/)
- Zhong Zhuang, zhuan143@umn.edu, [https://scholar.google.com/citations?user=rGGxUQEAAAAJ](https://scholar.google.com/citations?user=rGGxUQEAAAAJ)
- Hengyue Liang, liang656@umn.edu, [https://hengyuel.github.io/](https://hengyuel.github.io/)
- Le Peng, peng0347@umn.edu, [https://sites.google.com/view/le-peng/](https://sites.google.com/view/le-peng/)
- Hengkang Wang, wang9881@umn.edu, [https://www.linkedin.com/in/hengkang-henry-wang-a1b293104/](https://www.linkedin.com/in/hengkang-henry-wang-a1b293104/)
- Ju Sun, jusun@umn.edu, [https://sunju.org/](https://sunju.org/)
