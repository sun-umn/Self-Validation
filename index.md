## Abstract

Recent works have shown the surprising effectiveness of deep generative models in solving numerous image reconstruction (IR) tasks, ***without the need for any training set***. We call these models, such as deep image prior and deep decoder, collectively as ***single-instance deep generative priors*** (SIDGPs). However, often the successes hinge on appropriate early stopping, which by far has largely been handled in an ad hoc manner or even by visual inspection. In this paper, we propose the first principled method for early stopping when applying SIDGPs to image reconstruction, taking advantage of the typical bell trend of the reconstruction quality. In particular, our method is based on collaborative training and ***self-validation***: the primal reconstruction process is monitored by a deep autoencoder, which is trained online with the historic reconstructed images and used to validate the reconstruction quality constantly. On several IR problems and different SIDGPs that we experiment with, our self-validation method is able to reliably detect near-peak performance levels and signal good stopping points.


### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/sun-umn/Self-Validation/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
