import json
import os
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import pickle
import argparse

# Get entity names from FastRDFStore
# https://github.com/microsoft/FastRDFStore

from struct import *


class BinaryStream:
    def __init__(self, base_stream):
        self.base_stream = base_stream

    def readByte(self):
        return self.base_stream.read(1)

    def readBytes(self, length):
        return self.base_stream.read(length)

    def readChar(self):
        return self.unpack('b')

    def readUChar(self):
        return self.unpack('B')

    def readBool(self):
        return self.unpack('?')

    def readInt16(self):
        return self.unpack('h', 2)

    def readUInt16(self):
        return self.unpack('H', 2)

    def readInt32(self):
        return self.unpack('i', 4)

    def readUInt32(self):
        return self.unpack('I', 4)

    def readInt64(self):
        return self.unpack('q', 8)

    def readUInt64(self):
        return self.unpack('Q', 8)

    def readFloat(self):
        return self.unpack('f', 4)

    def readDouble(self):
        return self.unpack('d', 8)

    def decode_from_7bit(self):
        """
        Decode 7-bit encoded int from str data
        """
        result = 0
        index = 0
        while True:
            byte_value = self.readUChar()
            result |= (byte_value & 0x7f) << (7 * index)
            if byte_value & 0x80 == 0:
                break
            index += 1
        return result

    def readString(self):
        length = self.decode_from_7bit()
        return self.unpack(str(length) + 's', length)

    def writeBytes(self, value):
        self.base_stream.write(value)

    def writeChar(self, value):
        self.pack('c', value)

    def writeUChar(self, value):
        self.pack('C', value)

    def writeBool(self, value):
        self.pack('?', value)

    def writeInt16(self, value):
        self.pack('h', value)

    def writeUInt16(self, value):
        self.pack('H', value)

    def writeInt32(self, value):
        self.pack('i', value)

    def writeUInt32(self, value):
        self.pack('I', value)

    def writeInt64(self, value):
        self.pack('q', value)

    def writeUInt64(self, value):
        self.pack('Q', value)

    def writeFloat(self, value):
        self.pack('f', value)

    def writeDouble(self, value):
        self.pack('d', value)

    def writeString(self, value):
        length = len(value)
        self.writeUInt16(length)
        self.pack(str(length) + 's', value)

    def pack(self, fmt, data):
        return self.writeBytes(pack(fmt, data))

    def unpack(self, fmt, length = 1):
        return unpack(fmt, self.readBytes(length))[0]


def get_key(subject):
    if subject.startswith("m.") or subject.startswith("g."):
        if len(subject) > 3:
            return subject[0:4]
        elif len(subject) > 2:
            return subject[0:3]
        else:
            return subject[0:2]
    else:
        if len(subject) > 1:
            return subject[0:2]
        return subject[0:1]


def is_cvt(subject):
    tp_key = get_key(subject)
    if tp_key in cvt_nodes:
        if subject in cvt_nodes[tp_key]:
            return cvt_nodes[tp_key][subject]
    return False

def is_ent(tp_str):
    if len(tp_str) < 3:
        return False
    if tp_str.startswith("m.") or tp_str.startswith("g."):
        # print(tp_str)
        return True
    return False

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ent_input_path', required=True,
                        help='path of the extracted subgraph data')
    parser.add_argument('--output_path', required=True)

    args = parser.parse_args()

    print("Start extracting the mid names.")
    return args


if __name__ == '__main__':
    args = _parse_args()
    output_path = args.output_path
    ent_input_path = args.ent_input_path
    with open(ent_input_path, 'r') as f:
        all_ent = [line.strip().strip("\n") for line in f.readlines()]
    print("Load %d entities" % len(all_ent))

    # load mapping
    DATA_DIR = "/mnt/jiangjinhao/PLM4KBQA/data/Freebase"
    ALL_ENTITY_NAME_BIN = os.path.join(DATA_DIR, "FastRDFStore_data", "namesTable.bin")
    entity_names = {}
    with open(ALL_ENTITY_NAME_BIN, 'rb') as inf:
        stream = BinaryStream(inf)
        dict_cnt = stream.readInt32()
        print("total entities:", dict_cnt)
        for _ in range(dict_cnt):
            key = stream.readString().decode()
            # if key.startswith('m.') or key.startswith('g.'):
            # key = '/' + key[0] + '/' + key[2:]
            # if not key.startswith('m.') and not key.startswith('g.'):
            #     print("Key: %s"%(key))
            value = stream.readString().decode()
            entity_names[key] = value

    ALL_CVT_NAME_BIN = os.path.join(DATA_DIR, "FastRDFStore_data", "cvtnodes.bin")
    with open(ALL_CVT_NAME_BIN, 'rb') as cvtf:
        reader = BinaryStream(cvtf)
        dictionariesCount = reader.readInt32()
        print("total cvt entities:", dictionariesCount)
        cvt_nodes = {}
        for i in range(0, dictionariesCount):
            key = bytes.decode(reader.readString())
            # covert byte to string
            count = reader.readInt32()
            # print(key, count)
            dict_tp = {}
            for j in range(0, count):
                mid = bytes.decode(reader.readString())
                isCVT = reader.readBool()
                dict_tp[mid] = isCVT
            cvt_nodes[key] = dict_tp

    mid_subset = defaultdict()
    cvt_subset = defaultdict()
    no_mapping_subset = set()
    for ent in all_ent:
        if ent in entity_names:
            mid_subset[ent] = entity_names[ent]
        else:
            if is_ent(ent):
                no_mapping_subset.add(ent)
        cvt_subset[ent] = is_cvt(ent)

    print("There are %d--%d entities mapping or no mapping" % (len(mid_subset), len(no_mapping_subset)))
    data = (cvt_subset, mid_subset)
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
        print("Save the mapping dict to %s" % output_path)
