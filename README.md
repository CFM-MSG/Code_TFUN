# TFUN: Trilinear Fusion Network for Ternary Image-Text Retrieval

This repository is the Pytorch implementation of our work: *TFUN: Trilinear Fusion Network for Ternary Image-Text Retrieval* 

![image](https://github.com/CFM-MSG/Code_TFUN/blob/main/img/framework.png)

The proposed TFUN method contains three main procedures: 1) During feature embedding, the pre-trained ResNet-50, sentence2vector (S2V) and GRU are utilized to extract embedding of the image, instruction and ingredient respectively; 2) The cross-modal trilinear fusion consisting of attention mechanism and tensor decomposition is used for capturing the interaction between three inputs and learning the similarity score; 3) The similarity score of image-recipe pairs will be selected by our proposed three-stage hard triplet sampling strategy.

## Introduction

Recently, a special type of image-text retrieval task named *Ternary Image-Text Retrieval (TITR)* has drawn increasing attention. In this task, the total inputs of query and target consist of three components, rather than two in the widely-studied retrieval case. The typical TITR scenarios include recipe retrieval (e.g., ingredients text, instructions text and food images) and fashion search (\egs, original images,  text and modified images). 

![image](https://github.com/CFM-MSG/Code_TFUN/blob/main/img/retrieval_titr.png)

We propose a novel fusion framework named Trilinear FUsion Network (TFUN) for the TITR task, which utilizes high-level associations between these three inputs simultaneously and learn an accurate cross-modal similarity function via bi-directional triplet loss explicitly. To reduce the model complexity, we introduce the attention mechanism and tensor decomposition method which making the computation accessible. Furthermore, we also develop a three-stage hard triplet sampling scheme to ensure fast convergence and avoid model collapsed during training.

## Installation and Requirements

Install the dependencies with:

```shell
pip install -r requirements.txt
```

## Training and Testing

### Data Download

Recipe1M dataset: http://im2recipe.csail.mit.edu/dataset/download

### Training

```shell
python3 -u train.py
--img_path /img_path
--data_path /lmdb_path
--ingrW2V /vocab.bin_path
--dataset /lmdb_path
--log log/log_path
--snapshots /snapshots_path
--lr 1e-4 
```

* `--neg` : strategy of triplet sampling, default = 'semi', you can select 'semi', 'hard', 'ohnm' for the three-stage hard triplet sampling scheme.

### Testing

```shell
test.py
--img_path /img_path
--data_path /lmdb_path
--ingrW2V /vocab.bin_path
--dataset /lmdb_path
--resume /model_path
```

* `--val_num` : size of test set, default=1000.

## Supplementary Visual Results

### Recipe-to-Image retrieval results compared with ACME

We analyze the typical visualized results on recipe-to-image retrieval obtained from our TFUN method and the compared baseline ACME. The correct retrieved image is highlighted in red color.

![image](https://github.com/CFM-MSG/Code_TFUN/blob/main/img/comparision1.png)

![image](https://github.com/CFM-MSG/Code_TFUN/blob/main/img/comparision2.png)

### More examples of recipe-to-Image retrieval

Top-5 results of recipe-to-image retrieval obtained by our TFUN model. The correct retrieved image is highlighted in red color.

![image](https://github.com/CFM-MSG/Code_TFUN/blob/main/img/t2i_0.png)

![image](https://github.com/CFM-MSG/Code_TFUN/blob/main/img/t2i_1.png)

![image](https://github.com/CFM-MSG/Code_TFUN/blob/main/img/t2i_2.png)

![image](https://github.com/CFM-MSG/Code_TFUN/blob/main/img/t2i_3.png)

![image](https://github.com/CFM-MSG/Code_TFUN/blob/main/img/t2i_4.png)

![image](https://github.com/CFM-MSG/Code_TFUN/blob/main/img/t2i_fashion.png)

Scalability comparison of our TFUN model and the compared methods.

![image](https://github.com/CFM-MSG/Code_TFUN/blob/main/img/scalability_new.png)

### 



