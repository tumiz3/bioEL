"""
利用bios中获得的实体的同义词，构建sapbert训练需要的同义词对，构建的同义词对的形式为CID||synonym1||synonym2；
如果一个实体的同义词对的数量超过50，则从中随机选择50对。
"""
from tqdm import tqdm
import itertools
import random
import pickle

import sys
sys.path.append("../")
from utils import Logger

def get_synonyms(synonyms_path):
    with open(synonyms_path,"rb") as f:
        synonyms=pickle.load(f)
    return synonyms

def gen_pairs(input_list):
    #用于从一个实体的同义词中构建同义词对
    return list(itertools.combinations(input_list, r=2))

def gen_pos_pairs(synonyms):
    pos_pairs = []
    for k,v in tqdm(synonyms.items()):
        pairs = gen_pairs(v)
        if len(pairs)>50: # if >50 pairs, then trim to 50 pairs
            pairs = random.sample(pairs, 50)
        for p in pairs:
            line = str(k) + "||" + p[0] + "||" + p[1]
            pos_pairs.append(line)

    return pos_pairs

def main():
    with open("../data/")



