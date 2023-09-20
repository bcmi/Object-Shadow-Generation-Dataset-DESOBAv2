# Rendered-Shadow-Generation-Dataset-RdSOBA

**DESOBAv2** is a large-scale Shadow Generation dataset containing object-shadow pairs like [**DESOBA**](https://github.com/bcmi/Object-Shadow-Generation-Dataset-DESOBA) dataset with 28573 foreground objects, which is useful for supervised shadow generation methods.

<img src='examples/example_dataset.png' align="center" width=1024>

For more details, please refer to the following research paper:

> **DESOBAv2: owards Large-scale Real-world Dataset for Shadow Generation**  [[arXiv]](https://arxiv.org/pdf/2308.09972)<br>
>
> Qingyang Liu Tao†, Jianting Wang†, Li Niu

## Highlights

- 21573 images
- 28573 foreground objects
- nearly 36,000 pairs of augmented object-shadow pairs
- accurate masks
- diverse and realistic outdoor scenes
- multiplex viewpoints

## Downloads
We have provided some sample data. The relevant links are as follows: [[Baidu_Cloud]](https://pan.baidu.com/s/1akR8josre7Q1znQHUCjOzg ) (access code: bcmi)

If you are interested in our full dataset, please feel free to contact us via [Email](narumimaria@sjtu.edu.cn).

## Details

### Dataset Construction
We harvest an extensive collection of real-world outdoor images with natural lighting across various scenes from the Internet and manually filter the collected images. Given a real image, we use pretrained object-shadow detection model [1] to predict object and shadow masks for object-shadow pairs. We obtain the union of all detected shadow masks as the inpainting mask and apply the pretrained inpainting model [2] to get a deshadowed image.

After inpainting, we manually filter the object-shadow pairs according to the following rules: 1) We remove the object-shadow pairs with low-quality object masks or shadow masks. 2) We remove those object-shadow pairs with generated shadows or noticeable artifacts in the shadow region.

### Composite Image Synthesis
Given a pair of real image $I_r$ and deshadowed image $I_d$, we randomly select a foreground object from valid instances and synthesize the composite image. We replace the shadow regions of the other objects in $I_d$ with the counterparts in $I_r$ to synthesize a composite image $I_c$, in which only the selected foreground object does not have shadow and all the other objects have shadows. We adopt the second approach. 

After inpainting, the pixel values in the background may be slightly changed, that is, the background of  $I_c$ could be slightly different from that of $I_r$. To ensure consistent background, we obtain the ground-truth target image $I_g$  by replacing the shadow regions of all objects in $I_d$ with the counterparts in $I_r$. Then, $I_c$ and $I_g$ form a pair of input composite image and ground-truth target image.

Given an image with $K$ detected instances, we use $M_{o,k}$ (resp., $M_{s,k}$) to denote the object (resp., shadow) mask of the $k$-th object. When choosing the $k$-th object as foreground object,  $M_{o,k}$ (resp., $M_{s,k}$) is 
the foreground object (resp., shadow) mask $M_{fo}$ (resp., $M_{fs}$). We can merge $\{M_{o,1}, \ldots, M_{o,k-1}, M_{o,k+1}, \ldots, M_{o,K}\}$ as the background object mask $M_{bo}$. Similarly,  we can merge $\{M_{s,1}, \ldots, M_{s,k-1}, M_{s,k+1}, \ldots, M_{s,K}\}$ as the background shadow mask $M_{bs}$. Up to now, we obtain a tuple in the form of $\{I_c,M_{fo},M_{fs},M_{bo},M_{bs},I_g\}$, which is consistent with the tuple format in DESOBA dataset.



[1]Wang, Tianyu, et al. "Instance shadow detection with a single-stage detector." *IEEE Transactions on Pattern Analysis and Machine Intelligence* 45.3 (2022): 3259-3273.

[2]Rombach, Robin, et al. "High-resolution image synthesis with latent diffusion models." *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*. 2022.
