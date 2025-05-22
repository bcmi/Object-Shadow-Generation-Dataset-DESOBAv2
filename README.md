# Real-world Shadow Generation Dataset DESOBAv2 and the Official Implementation of SGDiffusion

This is the official repository for the following paper:

> **Shadow Generation for Composite Image Using Diffusion Model**  [[arXiv]](https://arxiv.org/pdf/2403.15234.pdf)<br>
>
> Qingyang Liu, Junqi You, Jianting Wang, Xinhao Tao, Bo Zhang, Li Niu<br>
> Accepted by **CVPR 2024**.

Our improved version GPSDiffusion can be found [here](https://github.com/bcmi/GPSDiffusion-Object-Shadow-Generation).


## Dataset


**DESOBAv2** is a large-scale real-world shadow generation dataset containing object-shadow pairs like [**DESOBA**](https://github.com/bcmi/Object-Shadow-Generation-Dataset-DESOBA) dataset with 28,573 foreground objects, which is useful for supervised shadow generation methods. It has 21,575 real images with 28,573 object-shadow pairs from outdoor scenes. In the following example images, from left to right, we show the composite image, the foreground object mask, the foreground shadow mask, the background object mask, the background shadow mask, the ground-truth target image.  


<img src='examples/example_dataset.png' align="center" width=70%>

You can download the full DESOBAv2 Dataset from [[Baidu_Cloud]](https://pan.baidu.com/s/1_nXb3ElxImmsq2BPcBGdPQ?pwd=bcmi) (access code: bcmi) or  [[Dropbox]](https://www.dropbox.com/scl/fo/f71dg98aszqxtn2qs3l1c/ALS7dpAe3dBPbYbRaq10mnY?rlkey=6cm1vcma91yn06ziy3v4cxzxg&st=pmc7niq3&dl=0). We release two versions: the full-resolution version and 256x256 version. 

We also release 22469 unused shadow images  [[Baidu_Cloud]](https://pan.baidu.com/s/1jfRADN2HI2YjL4L7y1JhmA) (access code: bcmi) or
 [[Dropbox]](https://www.dropbox.com/scl/fi/wi88s1mitkchvwqjmxd2y/unuse_img.rar?rlkey=7qc9e180betekchkunsuniuwj&st=gav0sech&dl=0), in case that you want to extend DESOBAv2 dataset. 

## Our SGDiffusion
Here we provide PyTorch implementation and the trained model of our SGDiffusion.

### Installation
- Clone this repo:
    git clone https://github.com/bcmi/Object-Shadow-Generation-Dataset-DESOBAv2.git
- Download the DESOBAv2 dataset from [[Baidu Cloud]](https://pan.baidu.com/s/1_nXb3ElxImmsq2BPcBGdPQ?pwd=bcmi) (access code: bcmi) or [[Dropbox]](https://www.dropbox.com/scl/fo/f71dg98aszqxtn2qs3l1c/ALS7dpAe3dBPbYbRaq10mnY?rlkey=6cm1vcma91yn06ziy3v4cxzxg&st=j5hx8as9&dl=0). Unzip `desobav2-256x256.rar` to `./data/`, and rename it to `desobav2`.
- Download the checkpoints from [[Baidu Cloud]](https://pan.baidu.com/s/11tr_H9YYIM7UzwOtbHzmpQ?pwd=bcmi) (access code: bcmi) or [[Dropbox]](https://www.dropbox.com/scl/fo/y345ctv86zdyamwxv4i4h/AGUqbGKxhD-ipBssbgXKrFw?rlkey=j7hieqaexadn18rbajkbzloj4&st=9xyk2yfa&dl=0). Unzip `ckpt.rar` to `./data/`. Note that we also provide an alternative model
`DESOBAv2_2.pth`, which can replace `DESOBAv2.pth` in `ckpt.rar`.

### Environment
    conda env create -f environment.yaml
    conda activate SGDiffusion

### Training
    python train_SGDiffusion.py

### Inference
    python infer_SGDiffusion.py

### Post-processing
    python post_processing.py

### Evaluation
    python eval_result.py
    
## Other Resources

+ [Awesome-Object-Shadow-Generation](https://github.com/bcmi/Awesome-Object-Shadow-Generation)
+ [Awesome-Image-Composition](https://github.com/bcmi/Awesome-Object-Insertion)
