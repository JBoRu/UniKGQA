### Preparing the datasets
Besides the origin datasets , we also leverage the preprocessed datasets from [NSM](https://github.com/RichardHGL/WSDM2021_NSM).
You should first download these datasets from [here]().
After downloading, you should unzip it and move the files to *data/*.
Then, you can perform the above command to prepare the dataset:

    cd dataset/
    python convert_webqsp_to_unify_format.py

**Note: some paths name may be modified according to your situation. We strongly recommend downloading our pre-processed data from [here]().**

### Preparing the KG
For WebQSP and CWQ, we use the whole freebase as the knowledge base.
We strongly suggest that you follow Freebase-Setup to build a Virtuoso for the Freebase dataset.
Following NSM and GraftNet, to improve the data accessing efficiency when training, we extract a 2-hop topic-centric subgraph for each question in WebQSP and a 4-hop topic-centric subgraph for each question in CWQ to dump the necessary facts.
Then, we access these necessary facts efficiently with sparse matrix multiplication based on the publicly released code of SubgraphRetrieval.
The following scripts aims to extract topic-centric subgraphs from the original KG and convert them to sparse KG.
    
    cd freebase/
    python extract_samples.py
    python get_seed_set.py
    get_2hop_subgraph.py
    convert_subgraph_to_int.py
    build_ent_type_ary.py

**Note: some paths name may be modified according to your situation. It is time-consuming, and we strongly recommend downloading our pre-processed data from [here]().**
