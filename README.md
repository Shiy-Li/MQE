# MQE
 This repository is the official implementation of "[Noise-Resilient Unsupervised Graph Representation Learning via Multi-Hop Feature Quality Estimation](https://arxiv.org/pdf/2407.19944)", accepted by CIKM 2024.


# Training on Colab
Click the ["Open in Colab"](https://colab.research.google.com/drive/1x5ln4NYgyIOoiPg-24UKHxmTH_USqK-c?usp=sharing) button to quickly reproduce the results in the Google Colab environment.

# Setup
conda create -n MQE python=3.8<br/>
conda activate MQE<br/>
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121<br/>
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-2.1.2+cu121.html<br/>
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-2.1.2+cu121.html<br/>
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-2.1.2+cu121.html<br/>
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-2.1.2+cu121.html<br/>
pip install torch-geometric

# Cite
If you compare with, build on, or use aspects of this work, please cite the following:

```js/java/c#/text
@article{li2024noise,
  title={Noise-Resilient Unsupervised Graph Representation Learning via Multi-Hop Feature Quality Estimation},
  author={Li, Shiyuan and Liu, Yixin and Chen, Qingfeng and Webb, Geoffrey I and Pan, Shirui},
  journal={arXiv preprint arXiv:2407.19944},
  year={2024}
}
```

