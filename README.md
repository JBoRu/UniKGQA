# UniKGQA
This is the official PyTorch implementation for the paper:
> UniKGQA: Unified Retrieval and Reasoning for Solving Multi-hop Question Answering Over Knowledge Graph - [ICLR 2022](https://arxiv.org/pdf/2212.00959.pdf)  
> Jinhao Jiang*, Kun Zhou*, Wayne Xin Zhao, Ji-Rong Wen

*Updates:*
1. - [x] Update camera-ready code repo!
2. - [ ] Update camera-ready paper!
3. - [ ] Upload our results to the leaderboard!

## Overview
We propose **UniKGQA**, which stands for **Uni**fied retrieval and reasoning for solving multi-hop **Q**uestion **A**nswering over **K**nowledge **G**raph.
It's a unified model architecture based on pre-trained language models (PLMs) for both retrieval and reasoning stages, 
which consists of the ***Semantic Matching*** module and the ***Matching Information Propagation*** module.
Furthermore, we design an effective learning strategy with both pre-training (i.e., ***Question-Relation Matchin***) and
fine-tuning (i.e., ***Retrieval on Abstract Subgraphs*** and ***Reasoning on Retrieved Subgraphs***) based on the unified architecture.
With this unified architecture, the proposed learning method can effectively enhance the sharing and transferring of relevance information between the two stages.

<p align="center">
  <img src="asset/model.png" alt="UniKGQA architecture" width="1000">
  <br>
  <b>Figure 1</b>: The overview of the unified model architecture of UniKGQA.
</p>

## Environments Setting
We implement our approach based on Pytorch and Huggingface Transformers.
We export the detailed environment settings in the freeze.yml, and you can install them with the command line as follows:

    conda env create -f freeze.yml

## Preparing the datasets
We preprocess the datasets following NSM.
You can see the *Preprocess* directory for details.

## Preparing the KG
For WebQSP and CWQ, we use the whole freebase as the knowledge base.
We strongly suggest that you follow Freebase-Setup to build a Virtuoso for the Freebase dataset.
Following NSM and GraftNet, to improve the data accessing efficiency when training, we extract a 2-hop topic-centric subgraph for each question in WebQSP and a 4-hop topic-centric subgraph for each question in CWQ to dump the necessary facts.
Then, we access these necessary facts efficiently with sparse matrix multiplication based on the publicly released code of SubgraphRetrieval.
You can see the *KnowledgeBase* directory for details.

**Note: these two processes are time-consuming, so we strongly suggest you download our preprocessed data and KG dump from [here]().**

## Pre-training
We utilize the shortest path between topic entities and answer entities in KG as weak supervision signals to pre-train the PLM for question-relation semantic matching.

### Preparing weakly supervised pre-training data
1. We should extract the shortest relation paths for each sample. 
2. We filter some relation paths whose answer precision is lower than a specified threshold (we use 0.1 here).  
3. We regard the relations in the above-extracted relation paths as positive relations and randomly select other relations as negative relations.
4. We can get the pre-training data. Each sample format is *(question, positive relations, negative relations)*.

We solve each step with a specific .py file and combine them into a .sh script that you can directly run to construct the pre-training data as follows:
    
    cd ./UniModel
    sh question_relation_pretrain_data_construction.sh

**Note: this process is time-consuming, so we strongly suggest you download our pre-processed training data from [here]().**  
Then unzip it and move the *data/* directory below *UniModel/*.

### Pre-training with Question-Relation Matching
Based on the publicly released code of SimCSE, we pre-train the PLM with contrastive learning as follows:
    
    cd ./UniModel
    sh question_relation_matching_pretrain.sh
    
**Note: we almost do not fine-tune any hyper-parameters, such as learning rate, temperature, and the number of negatives.
You can also directly download our pre-trained model checkpoint from [here]().**

## Fine-tuning
After pre-training the PLM, we initialize the PLM parameters of UniKGQA with the pre-trained checkpoint.
Then, we fix the PLM parameters and only fine-tune other parameters during the whole fine-tuning stage.

### Constructing the Abstract Subgraph from KG
First, we should construct the abstract subgraph from the original KG for each sample.
When we shrink the abstract subgraph, we can leverage the pre-trained PLM for filtering obviously irrelevant relations.
    
    cd ./UniModel
    sh extract_abstract_subgraph_from_kg.sh

**Note: this process is time-consuming, so we strongly suggest you download our pre-processed abstract subgraph data from [here]().**  
Then unzip it and move the *data/* directory below *UniModel/*.
    
### Fine-tuning for Retrieval on Abstract Subgraphs
Second, we fine-tune our UniKGQA reasoning model (UniKGQA-ret) on the abstract subgraph.
Note we initialize the PLM module of UniKGQA-ret with the pre-trained parameters and fix them during the fine-tuning.
You can train the UniKGQA-ret as follows:
    
    cd ./UniModel/nsm_retriever
    sh fine_tune_for_retrieval_on_abstract_subgraph.sh

**Note: we almost do not fine-tune any hyper-parameters, such as learning rate, temperature, and the number of negatives.
You can also directly download our fine-tuned model checkpoint on abstract subgraph from [here]().**

### Performing Retrieval with UniKGQA
After fine-tuning the UniKGQA-ret, you can leverage it to retrieve the subgraph for the latter reasoning stage as follows:
    
    cd ./UniModel
    sh perform_retrieval.sh

**Note: this process is time-consuming, so we strongly suggest you download our retrieved subgraph data from [here]().**  
Then unzip it and move the *data/* directory below *UniModel/*.

### Fine-tuning for Reasoning on Retrieved Subgraphs
After obtaining the retrieved subgraph, you can fine-tune the UniKGQA reasoning model (UniKGQA-rea).
Note we initialize the UniKGQA-rea with UniKGQA-ret to share the learned knowledge during the retrieval stage and still fix the parameters of the PLM module.
You can train the UniKGQA-ret as follows:
    
    cd ./UniModel
    sh perform_reasoning.sh 

**Note: we almost do not fine-tune any hyper-parameters, such as learning rate, temperature, and the number of negatives.
You can also directly download our fine-tuned model checkpoint on instantiated subgraph from [here]().**


## Acknowledge
Our code refers to the publicly released code of NSM, SimCSE, and SubgraphRetrieval.
Thanks for their contribution to the research.