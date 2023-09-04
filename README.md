# 3DDualDomainAttention
3D-DDA: 3D Dual-Domain Attention For Brain Tumor Segmentation

---

## Author

- [Nhu-Tai Do](https://dntai.vneasy.net/)

- [Hoang Son-Vo Thanh](https://sonvth.vercel.app/about)

- [Tram-Tran Nguyen-Quynh]()

- [Soo-Hyung Kim]()

---

- [Slide]

- [Paper]

- [Processdings]

---

# Abstract


<div align="center">
<img width=500 src= "static/fig1_demo_(1).png"/>
<p align="center"> Grad-cam visualization of the encoding feature map at three axes in DynUnet with/without 3D-DDA </p>
</div>

Accurate brain tumor segmentation plays an essential role in the diagnosis process. However, there are challenges due to the variety of tumors in low contrast, morphology, location, annotation bias, and imbalance among tumor regions. This work proposes a novel 3D dual-domain attention module to learn local and global information in spatial and context domains from encoding feature maps in Unet. Our attention module generates refined feature maps from the enlarged reception field at every stage by attention mechanisms and residual learning to focus on complex tumor regions. Our experiments on BraTS 2018 have demonstrated superior performance compared to existing state-of-the-art methods

---
# Method

<div align="center">
<img width=500 src= "static/fig2_overview.png"/>
<p> 3D Dual-domain Attention attached into DynUnet backbone at four stages </p>
</div>

**More detail…**

<div align="center">
<img width=500 src= "static/fig3_details.png"/>
<p> 3D-DDA block details. </p>
</div>

# Paper
<div align="center">
<img width=500 src= "static/067757__80685.png" href="https://1drv.ms/b/s!ArlplJhiPYx6gnw8_jQHEYPu2_sc?e=dSPoOn"/>
<p href="https://1drv.ms/b/s!ArlplJhiPYx6gnw8_jQHEYPu2_sc?e=dSPoOn">[PDF]</p>
</div>


# Citation

```
@INPROCEEDINGS{,
  author={Nhu-Tai Do, Hoang-Son Vo-Thanh, Tram-Tran Nguyen-Quynh, Soo-Hyung Kim},
  booktitle={2023 IEEE International Conference on Image Processing (ICIP)}, 
  title={3D-DDA: 3D Dual-Domain Attention For Brain Tumor Segmentation}, 
  year={2023},
  volume={},
  number={},
  pages={},
  doi={}}
```