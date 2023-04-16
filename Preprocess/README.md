### Preparing the datasets
You should first download the original datasets from [here](https://www.microsoft.com/en-us/download/details.aspx?id=52763)
After downloading, you should unzip it and move the files to *data/webqsp/webqsp_ori/*.
Besides the origin datasets , we also leverage the preprocessed datasets from [NSM](https://drive.google.com/drive/folders/1qRXeuoL-ArQY7pJFnMpNnBu0G-cOz6xv).
After downloading, you should unzip it and move the files to *data/webqsp/webqsp_NSM/*.
The directory structure should be:

- data
    - webqsp
        - webqsp_NSM
            - dev_simple.json
            - test_simple.json
            - train_simple.json
            - ...
        - webqsp_ori
            - WebQSP.train.json
            - WebQSP.test.json
            - ...

Then, you can perform the follow command to prepare the dataset:

    cd dataset/
    python convert_webqsp_to_unify_format.py

**Note: some paths name may be modified according to your situation. We strongly recommend downloading our pre-processed data from [here](https://drive.google.com/file/d/1plbFei-BvtcurgTNhHcrjD1N71t_5LVT/view?usp=share_link).**
After downloading, you should unzip it and move the files to *data/*.

### Preparing the KG
For WebQSP and CWQ, we use the whole freebase as the knowledge base.
We strongly suggest that you follow Freebase-Setup to build a Virtuoso for the Freebase dataset.
Following NSM and GraftNet, to improve the data accessing efficiency when training, we extract a 2-hop topic-centric subgraph for each question in WebQSP and a 4-hop topic-centric subgraph for each question in CWQ to dump the necessary facts.
Then, we access these necessary facts efficiently with sparse matrix multiplication based on the publicly released code of SubgraphRetrieval.
The following scripts aims to extract topic-centric subgraphs from the original KG and convert them to sparse KG.
    
    cd freebase/
    sh ./run_process.sh

**Note: some paths name may be modified according to your situation. It is time-consuming, and we strongly recommend downloading our pre-processed data from [here]().**
