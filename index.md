<div align="center">
<figure><img src="figures/framework.png" width="800"></figure>
 <br>
</div>



## Abstract

Recent works have shown the surprising effectiveness of deep generative models in solving numerous image reconstruction (IR) tasks, ***without the need for any training set***. We call these models, such as deep image prior and deep decoder, collectively as ***single-instance deep generative priors*** (SIDGPs). However, often the successes hinge on appropriate early stopping, which by far has largely been handled in an ad hoc manner or even by visual inspection. 

In this paper, we propose the first principled method for early stopping when applying SIDGPs to image reconstruction, taking advantage of the typical bell trend of the reconstruction quality. In particular, our method is based on collaborative training and ***self-validation***: the primal reconstruction process is monitored by a deep autoencoder, which is trained online with the historic reconstructed images and used to validate the reconstruction quality constantly. On several IR problems and different SIDGPs that we experiment with, our self-validation method is able to reliably detect near-peak performance levels and signal good stopping points.


## Early stopping via self-validation



## Image denoising
TBD

## MRI reconstruction
TBD

## Image regression
TBD

## Contact
TBD
