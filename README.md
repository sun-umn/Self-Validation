# Self-Validation

## Self-Validation: Early Stopping for Single-Instance Deep Generative Priors

This is the official implementation of our paper *Self-Validation: Early Stopping for Single-Instance Deep Generative Priors* which has been accepted by the [32nd British Machine Vision Conference 2021](http://www.bmvc2021.com/). You can find our paper via [this link](http://) and the project webpage via [this link](https://sun-umn.github.io/Self-Validation/).


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
We provide a ready-to-use dataset under the folder `/Dataset` where there are 4 types of noises we used in our paper.

Alternatively, you can also create the dataset by yourself. The clean images can be downloaded [here](https://webpages.tuni.fi/foi/GCF-BM3D/index.html#ref_results). After you have the clean images, you can follow the [ImageNet-C protocol](https://github.com/hendrycks/robustness) (or write corruption functions by yourself) to create the corrupted images. For the parameters for each noise level, please check the Appendix of [our paper](http://).

### DIP+AE


### DD+AE

## Citation/BibTex

More technical details and experiemntal results can be found in our paper:

Taihui Li, Zhong Zhuang, Hengyue Liang, Le Peng, Hengkang Wang, Ju Sun. Self-Validation: Early Stopping for Single-Instance Deep Generative Priors. 32nd British Machine Vision Conference 2021.

## Contact
- Taihui Li, lixx5027@umn.edu, [https://taihui.github.io/](https://taihui.github.io/)
- Zhong Zhuang, zhuan143@umn.edu, [https://scholar.google.com/citations?user=rGGxUQEAAAAJ](https://scholar.google.com/citations?user=rGGxUQEAAAAJ)
- Hengyue Liang, liang656@umn.edu, [https://hengyuel.github.io/](https://hengyuel.github.io/)
- Le Peng, peng0347@umn.edu, [https://sites.google.com/view/le-peng/](https://sites.google.com/view/le-peng/)
- Hengkang Wang, wang9881@umn.edu, [https://www.linkedin.com/in/hengkang-henry-wang-a1b293104/](https://www.linkedin.com/in/hengkang-henry-wang-a1b293104/)
- Ju Sun, jusun@umn.edu, [https://sunju.org/](https://sunju.org/)
