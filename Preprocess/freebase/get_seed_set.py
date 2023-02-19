import json
import sys


input_file = "/mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/cwq/cwq_step0.json"
output_file = "/mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/cwq/cwq_seed.txt"
# input_file = "/mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/webqsp/webqsp_step0.json"
# output_file = "/mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/webqsp/webqsp_seed.txt"
# input_file = sys.argv[1]
# output_file = sys.argv[2]
f = open(input_file)
f1 = open(output_file, "w")
entity_set = set()
for line in f:
    q_obj = json.loads(line)
    entity_list = q_obj["entities"]
    for entity in entity_list:
        entity_set.add(entity["kb_id"])
f.close()
for entity in entity_set:
    f1.write("%s\n" % entity)
f1.close()
