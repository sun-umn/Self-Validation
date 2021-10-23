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

4. Create a new conda environment with the YML file we provided:

    `conda env create -f environment.yml`
   
5.  Activate conda environment and you are now ready to explpre the codes/models!
    
    `conda activate pytorch_py3.6`
    
## Explore the Codes/Models

### Dataset
We provide a ready-to-use dataset under the folder `/Dataset` where there are 4 types of noises we used in our paper.
