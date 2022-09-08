# TFUN: Trilinear Fusion Network for Ternary Image-Text Retrieval

This repository is the Pytorch implementation of our work: *TFUN: Trilinear Fusion Network for Ternary Image-Text Retrieval* 

![image](https://github.com/CFM-MSG/Code_TFUN/blob/main/img/framework.png)

The proposed TFUN method contains three main procedures: 1) During feature embedding, the pre-trained image encoder and text encoder are utilized to extract embedding of three inputs respectively; 2) The trilinear fusion consisting of attention mechanism and tensor decomposition is used for capturing the interaction between three inputs and earning the similarity score; 3) The similarity score of triplet pairs will be selected by our proposed three-stage hard triplet sampling strategy.

## Introduction

Recently, a particular type of image-text retrieval task named *Ternary Image* *Text Retrieval* (TITR) has drawn increasing attention. In this task, the total inputs of query and target consist of three components, rather than two inputs in the widely-studied retrieval case. The typical TITR scenarios include recipe retrieval (*e.g.*, ingredients text, instructions text and food images) and fashion search (*e.g.*, original images, text and modifified images). 

![image](https://github.com/CFM-MSG/Code_TFUN/blob/main/img/retrieval_titr.png)

we propose a novel fusion framework named **T***rilinear* **FU***sion* **N***etwork (TFUN)* to utilize high-level associations between these three inputs simultaneously and learn an accurate cross-modal similarity function via bi-directional triplet loss explicitly, which is generic for the TITR task. To reduce the model complexity, we introduce the advanced method of tensor decomposition to ensure computational efficiency and accessibility. 

![image](https://github.com/CFM-MSG/Code_TFUN/blob/main/img/triplet.png)

We also propose a three-stage hard negative sampling scheme to use all anchor-positive pairs and only concentrate on mining the hard negative samples. The triplet diagram on the left plots a sample as a dot, the distance between two dots represents the similarity between them. We take an example from the recipe-to-image retrieval task on the right side of the fifigure. 



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

## Supplementary Results

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

### The effect of hard negative sampling strategy

![image](https://github.com/CFM-MSG/Code_TFUN/blob/main/img/comparision2.png)

The effect of the hard negative sampling and the potential ability of our proposed model can be indicated in Table 5. The framework of the TFUN model is specially designed for the TITR task so it is more complex and difficult to train compared to traditional bilinear models. Our TFUN model outperforms some of the traditional methods when trained with basic semi-hard negative samples. Nevertheless, it achieves state-of-the-arts retrieval performance when trained with the proposed three-stage hard negative sampling scheme, which consistently demonstrates the capability of our entire proposed model.

### The effect of adopting BERT as the text encoder instead of LSTM.

![image](https://github.com/CFM-MSG/Code_TFUN/blob/main/img/comparision2.png)

We have conducted additional ablation study to check whether our proposed method can take benefits by using pre-trained text encoder, e.g. BERT.

The results shows that the retrieval performance of our proposed TFUN model improved after adopting BERT in both recipe retrieval task and fashion search task, which show good scalability of our model.
We believe that's because BERT can capture more latent semantic information.

However, since our TFUN model focuses on the fusion strategy between three inputs, we only use LSTM as the text encoder in the main paper following most of the prior work.



