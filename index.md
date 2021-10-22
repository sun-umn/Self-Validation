<div align="center">
<figure><img src="figures/framework.png" width="800"></figure>
 <br>
</div>



## Early stopping via self-validation

Recent works have shown the surprising effectiveness of deep generative models in solving numerous image reconstruction (IR) tasks, ***without the need for any training set***. We call these models, such as deep image prior and deep decoder, collectively as ***single-instance deep generative priors*** (SIDGPs). However, often the successes hinge on appropriate early stopping (see [Figure 1](http://)), which by far has largely been handled in an ad hoc manner or even by visual inspection. 

<div align="center">
<figure><img src="figures/dip_dd_comb-01.png" width="800"></figure>
 <br>
 <figcaption>Figure 1: Illustration of the overfitting issue of DIP and DD on image denoising with Gaussian noise (noise level: σ = 0.18; pixel values normalized to [0, 1]). The reconstruction quality (as measured by both PSNR and SSIM) typically follows a skewed bell curve: before the peak only the clean image content is recovered, but after the peak noise starts to kick in. Note that DD is not free from overfitting when the network is increasingly over-parametrized (indicated by the number following “DD-” in the legend).</figcaption>
</div>

In this paper, we propose the first principled method for early stopping when applying SIDGPs to image reconstruction, taking advantage of the typical bell trend of the reconstruction quality. In particular, our method is based on collaborative training and ***self-validation***: the primal reconstruction process is monitored by a deep autoencoder, which is trained online with the historic reconstructed images and used to validate the reconstruction quality constantly. On several IR problems and different SIDGPs that we experiment with, our self-validation method is able to reliably detect near-peak performance levels and signal good stopping points.





## Image denoising
TBD

## MRI reconstruction
TBD

## Image regression
TBD

## Contact
TBD
