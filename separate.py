import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import random_split
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
MAX_LEN = 20

def main():
    print("Source language - Target language: ")
    src_lng, trg_lng = input().split('-')

    lines = open('data/%s-%s.txt' % (src_lng, trg_lng), encoding='utf-8').read().strip().split('\n')
    random.shuffle(lines)

    if not os.path.exists('data/src'):
        os.makedirs('data/src')
    if not os.path.exists('data/trg'):
        os.makedirs('data/trg')
    
    phrases = {
        'trg' : [],
        'src' : [],
    }

    for line in lines:
        line_split = line.split('\t')
        if len(line_split[0].split(' ')) > MAX_LEN or len(line_split[1].split(' ')) > MAX_LEN:
            continue 
        phrases["src"].append(line_split[0])
        phrases["trg"].append(line_split[1])
    
    train_len = int(len(phrases['src'])*0.8)
    test_len = int(len(phrases['src'])*0.1)

    for t in ['src','trg']:
        f = open('data/{0}/{0}-train.txt'.format(t),'w',encoding='utf-8')
        for phrase in phrases[t][:train_len]:
            f.write(phrase + '\n')
        f = open('data/{0}/{0}-val.txt'.format(t),'w',encoding='utf-8')
        for phrase in phrases[t][train_len:-test_len]:
            f.write(phrase + '\n')
        f = open('data/{0}/{0}-test.txt'.format(t),'w',encoding='utf-8')
        for phrase in phrases[t][-test_len:]:
            f.write(phrase + '\n')


if __name__ == "__main__":
    main()