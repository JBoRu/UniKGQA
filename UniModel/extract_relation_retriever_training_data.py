import json
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_qid_path', required=True,
                        help='the input data path used for extracting the shortest paths')
    parser.add_argument('--pn_pairs_path', required=True,
                        help='the input data path used for extracting the shortest paths')
    parser.add_argument('--output_path', required=True,
                        help='the output data path used for extracting the shortest paths')
    parser.add_argument('--split_list', nargs="+")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()

    pn_pairs_path = args.pn_pairs_path
    with open(pn_pairs_path, "r") as f:
        all_pn_pair_data = f.readlines()
        all_pn_pair_data = [json.loads(l) for l in all_pn_pair_data]

    for split in args.split_list:
        print("Start extract %s set."%(split))
        out_path = args.output_path.replace("SPLIT", split)
        split_qid_path = args.split_qid_path.replace("SPLIT", split)
        split_qids = np.load(split_qid_path)
        all_samples = []
        for pair in tqdm(all_pn_pair_data, total=len(all_pn_pair_data)):
            qid = pair["ID"]
            if qid in split_qids:
                if "PosNegPairs" not in pair:
                    print(pair.keys())
                pn_pairs = pair["PosNegPairs"]
                all_samples.extend(pn_pairs)
        test = pd.DataFrame(data=all_samples)
        test.to_csv(out_path, index=False, header=True)
        print("Extract %d PN pairs and dump to %s." % (len(all_samples), out_path))

