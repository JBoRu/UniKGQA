# webqsp
#python ./manual_filter_rel.py /mnt/jiangjinhao/PLM4KBQA/data/Freebase/FastRDFStore_data/fb_en.txt /mnt/jiangjinhao/PLM4KBQA/freebase/manual_fb_filter.txt
#python ./extract_samples.py 0 /mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/webqsp /mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/webqsp/webqsp_step0_test.json "['test.jsonl']"
#python get_seed_set.py /mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/webqsp/webqsp_step0_test.json /mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/webqsp/webqsp_seed_test.txt
#python get_2hop_subgraph.py /mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/webqsp/webqsp_seed_test.txt /mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/webqsp/subgraph_test/ /mnt/jiangjinhao/PLM4KBQA/freebase/manual_fb_filter.txt
#python convert_subgraph_to_int.py /mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/webqsp/subgraph_test/  /mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/webqsp/subgraph_test/subgraph_hop2.txt  /mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/webqsp/subgraph_test/relations.txt /mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/webqsp/subgraph_test/entities.txt
#python build_ent_type_ary.py /mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/webqsp/subgraph_test/
#python extract_mid_names.py --ent_input_path /mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/webqsp/subgraph_test/entities.txt --output_path /mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/webqsp/subgraph_test/entity_name.pickle

# cwq
#python ./extract_samples.py 0 /home/jiangjinhao/work/QA/UniKBQA/UniModel/data/cwq "['test.jsonl']" /mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/cwq/cwq_step0_test.json
#python get_seed_set.py /mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/cwq/cwq_step0_test.json /mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/cwq/cwq_seed_test.txt
#python get_2hop_subgraph.py /mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/cwq/cwq_seed_test.txt /mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/cwq/subgraph_test/ /mnt/jiangjinhao/PLM4KBQA/freebase/manual_fb_filter.txt
#python convert_subgraph_to_int.py /mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/cwq/subgraph_test/  /mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/cwq/subgraph_test/subgraph_hop2.txt  /mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/cwq/subgraph_test/relations.txt /mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/cwq/subgraph_test/entities.txt
#python build_ent_type_ary.py /mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/cwq/subgraph_test/
#python extract_mid_names.py --ent_input_path /mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/cwq/subgraph_test/entities.txt --output_path /mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/cwq/subgraph_test/entity_name.pickle