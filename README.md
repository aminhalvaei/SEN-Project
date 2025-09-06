
# Extending RotatE: Interpretability and Multi-Relational Modeling
**Introduction**

This is an extended PyTorch implementation of the [RotatE](https://openreview.net/forum?id=HkgEQnRqYQ) model for knowledge graph embedding (KGE). We provide a toolkit that aims to improve performance of several popular KGE models. There is also some new models (RotateCT, MRotatE, MRotatECT) in addition to main models provided in the baseline. The toolkit has also used a new method for negative sampling phase that uses temprature annealing to make balance between the exploration and exploitation.

A faster multi-GPU implementation of RotatE and other KGE models is available in [GraphVite](https://github.com/DeepGraphLearning/graphvite).

This project is a fork of [Repository](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)
that belongs to RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space paper.

## **Execution Instructions**

A jupyter notebook is provided to make execution of model easier in google colab enviroment.
You can access it [here](https://github.com/aminhalvaei/SEN-Project/blob/master/SEN_Project.ipynb).

---

**Implemented features**

Models:
 - [x] RotatE
 - [x] pRotatE
 - [x] TransE
 - [x] ComplEx
 - [x] DistMult
 - [x] RotateCT (new)
 - [x] MRotatE (new)
 - [x] MRotatECT (new)

Evaluation Metrics:

 - [x] MRR, MR, HITS@1, HITS@3, HITS@10 (filtered)
 - [x] AUC-PR (for Countries data sets)

Loss Function:

 - [x] Uniform Negative Sampling
 - [x] Self-Adversarial Negative Sampling
 - [x] Temperature-Annealed Self-Adversarial Negative Sampling (new)

**Usage**

Knowledge Graph Data:
 - *entities.dict*: a dictionary map entities to unique ids
 - *relations.dict*: a dictionary map relations to unique ids
 - *train.txt*: the KGE model is trained to fit this data set
 - *valid.txt*: create a blank file if no validation data is available
 - *test.txt*: the KGE model is evaluated on this data set

**Reproducing the best results**

To reprocude the results in the ICLR 2019 paper [RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space](https://openreview.net/forum?id=HkgEQnRqYQ), you can run the bash commands in best_config.sh to get the best performance of RotatE, TransE, and ComplEx on five widely used datasets (FB15k, FB15k-237, wn18, wn18rr, Countries).

The run.sh script provides an easy way to search hyper-parameters:

    bash run.sh train RotatE FB15k 0 0 1024 256 1000 24.0 1.0 0.0001 150000 16 1.0 0 -de


**Speed**

The KGE models usually take about half an hour to run 10000 steps on a single GeForce GTX 1080 Ti GPU with default configuration. And these models need different max_steps to converge on different data sets:

| Dataset | FB15k | FB15k-237 | wn18 | wn18rr | Countries S* |
|-------------|-------------|-------------|-------------|-------------|-------------|
|MAX_STEPS| 150000 | 100000 | 80000 | 80000 | 40000 | 
|TIME| 9 h | 6 h | 4 h | 4 h | 2 h | 

**Results of the RotatE model**

| Dataset | FB15k | FB15k-237 | wn18 | wn18rr |
|-------------|-------------|-------------|-------------|-------------|
| MRR | .797 ± .001 | .337 ± .001 | .949 ± .000 |.477 ± .001
| MR | 40 | 177 | 309 | 3340 |
| HITS@1 | .746 | .241 | .944 | .428 |
| HITS@3 | .830 | .375 | .952 | .492 |
| HITS@10 | .884 | .533 | .959 | .571 |

**Using the library**

The python libarary is organized around 3 objects:

 - TrainDataset (dataloader.py): prepare data stream for training
 - TestDataSet (dataloader.py): prepare data stream for evluation
 - KGEModel (model.py): calculate triple score and provide train/test API

The run.py file contains the main function, which parses arguments, reads data, initilize the model and provides the training loop.

Add your own model to model.py like:
```
def TransE(self, head, relation, tail, mode):
    if mode == 'head-batch':
        score = head + (relation - tail)
    else:
        score = (head + relation) - tail

    score = self.gamma.item() - torch.norm(score, p=1, dim=2)
    return score
```

**Citation**

If you use the codes, please cite the following [paper](https://openreview.net/forum?id=HkgEQnRqYQ):

```
@inproceedings{
 sun2018rotate,
 title={RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space},
 author={Zhiqing Sun and Zhi-Hong Deng and Jian-Yun Nie and Jian Tang},
 booktitle={International Conference on Learning Representations},
 year={2019},
 url={https://openreview.net/forum?id=HkgEQnRqYQ},
}
```
