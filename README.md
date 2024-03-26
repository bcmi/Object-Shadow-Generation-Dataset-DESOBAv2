# Real-world Shadow Generation Dataset DESOBAv2

This is the official repository for the following paper:

> **Shadow Generation for Composite Image Using Diffusion Model**  [[arXiv]](https://arxiv.org/pdf/2403.15234.pdf)<br>
>
> Qingyang Liu, Junqi You, Jianting Wang, Xinhao Tao, Bo Zhang, Li Niu<br>
> Accepted by **CVPR 2024**.

**Our model has been integrated into our image composition toolbox libcom https://github.com/bcmi/libcom. Welcome to visit and try ＼(^▽^)／** 


## Dataset


**DESOBAv2** is a large-scale real-world shadow generation dataset containing object-shadow pairs like [**DESOBA**](https://github.com/bcmi/Object-Shadow-Generation-Dataset-DESOBA) dataset with 28,573 foreground objects, which is useful for supervised shadow generation methods. It has 21,575 real images with 28,573 object-shadow pairs from outdoor scenes. In the following example images, from left to right, we show the composite image, the foreground object mask, the foreground shadow mask, the background object mask, the background shadow mask, the ground-truth target image.  


<img src='examples/example_dataset.png' align="center" width=70%>

You can download the full DESOBAv2 Dataset from [[Baidu_Cloud]](https://pan.baidu.com/s/1_nXb3ElxImmsq2BPcBGdPQ?pwd=bcmi) (access code: bcmi) or [[One Drive]](https://1drv.ms/f/c/f4cc25a47574cccf/EmbMRAowxytJiM1KHeeeMqQBD4p1SyIShUdO2PZArkGOIA?e=sEd7ga). We release two versions: the full-resolution version and 256x256 version. 

We also release 22469 unused shadow images  [[Baidu_Cloud]](https://pan.baidu.com/s/1jfRADN2HI2YjL4L7y1JhmA) (access code: bcmi)
 [[One Drive]](https://1drv.ms/u/c/f4cc25a47574cccf/EYyhklCnf8ZPjzTXCDvXHjQBXwwvhLHzTeaD5B0Nf8w5Jg?e=kn48Vq), in case that you want to extend DESOBAv2 dataset. 



## Other Resources

+ [Awesome-Object-Shadow-Generation](https://github.com/bcmi/Awesome-Object-Shadow-Generation)
+ [Awesome-Image-Composition](https://github.com/bcmi/Awesome-Image-Composition)
