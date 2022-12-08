import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import cycle
from collections import defaultdict

# Cross-entropy loss/ No. of words
loss = {
    'average': [4.22, 2.34, 1.70, 1.36, 1.13, 0.99, 0.87, 0.79, 0.72, 0.67, 0.63, 0.59, 0.56, 0.53, 0.51, 0.49, 0.47, 0.46, 0.44, 0.42, 0.41, 0.40, 0.39, 0.38, 0.37, 0.36, 0.35, 0.35, 0.34, 0.33, 0.33, 0.32, 0.32, 0.31, 0.31, 0.30, 0.30, 0.29, 0.29, 0.28],
    'multihead': [4.16, 2.22, 1.62, 1.31, 1.10, 0.96, 0.85, 0.77, 0.70, 0.66, 0.62, 0.58, 0.55, 0.52, 0.50, 0.48, 0.46, 0.44, 0.43, 0.41, 0.40, 0.39, 0.38, 0.37, 0.36, 0.35, 0.34, 0.34, 0.33, 0.32, 0.32, 0.31, 0.31, 0.30, 0.30, 0.29, 0.28, 0.28, 0.28, 0.27],
    'bahdanau': [3.43, 2.07, 1.58, 1.33, 1.14, 1.00, 0.91, 0.83, 0.78, 0.74, 0.70, 0.67, 0.65, 0.62, 0.60, 0.57, 0.56, 0.54, 0.53, 0.52, 0.51, 0.50, 0.50, 0.49, 0.48, 0.47, 0.47, 0.46, 0.46, 0.45, 0.45, 0.44, 0.44, 0.43, 0.43, 0.43, 0.42, 0.42, 0.42, 0.42],
    'luong': [3.66, 2.42, 1.91, 1.66, 1.47, 1.37, 1.27, 1.22, 1.13, 1.09, 1.02, 1.05, 0.98, 0.95, 1.00, 0.98, 0.96, 1.09, 0.97, 0.96, 1.02, 0.96, 0.97, 1.00, 1.05, 1.09, 1.09, 1.17, 1.09, 1.17, 1.37, 1.75, 1.94, 2.22, 2.37, 2.54, 2.57, 2.68, 2.54, 2.44],
}
accuracy_test = {
    'average': [33.18, 56.31, 64.79, 69.46, 73.12, 75.62, 77.97, 79.97, 81.52, 82.75, 83.71, 84.73, 85.49, 86.24, 86.89, 87.40, 87.98, 88.34, 88.83, 89.24, 89.60, 89.94, 90.16, 90.55, 90.69, 90.99, 91.26, 91.41, 91.65, 91.83, 91.97, 92.18, 92.32, 92.51, 92.62, 92.73, 92.91, 93.02, 93.17, 93.24],
    'multihead':[34.19, 58.39, 66.24, 70.75, 74.06, 76.55, 78.83, 80.71, 82.18, 83.32, 84.27, 85.18, 85.92, 86.64, 87.24, 87.82, 88.34, 88.72, 89.18, 89.58, 89.88, 90.26, 90.48, 90.87, 91.04, 91.30, 91.61, 91.69, 91.98, 92.10, 92.32, 92.48, 92.58, 92.86, 92.88, 93.05, 93.25, 93.34, 93.46, 93.54],
    'bahdanau':[45.27, 61.43, 67.62, 71.03, 74.17, 76.61, 78.74, 80.51, 81.66, 82.73, 83.53, 84.34, 84.98, 85.63, 86.17, 86.59, 87.08, 87.34, 87.82, 88.07, 88.41, 88.65, 88.87, 89.15, 89.29, 89.53, 89.75, 89.93, 90.10, 90.20, 90.39, 90.43, 90.62, 90.86, 90.88, 90.98, 91.04, 91.24, 91.31, 91.32],
    'luong':[42.37, 56.05, 62.29, 65.48, 68.27, 69.66, 71.46, 72.49, 74.30, 75.14, 75.87, 76.74, 76.74, 76.05, 77.82, 78.46, 77.20, 77.84, 78.25, 75.15, 78.14, 78.31, 76.79, 78.38, 78.08, 77.38, 76.29, 75.41, 75.38, 73.68, 69.76, 62.96, 59.76, 55.45, 53.36, 50.20, 49.51, 47.74, 49.79, 51.03]
}
accuracy_valid = {
    'average': [62.251, 68.5106, 70.3323, 71.3386, 71.8776, 72.0823, 72.2502, 72.5063, 72.5989, 72.6117, 72.7496, 72.508, 72.8352, 72.902, 72.9865, 72.7737, 72.9507, 72.9378, 73.0303, 72.9207],
    'multihead': [63.8674, 69.7163, 71.2926, 72.008, 72.5203, 72.7785, 72.9801, 72.8518, 73.1009, 73.1715, 73.2929, 73.1159, 73.1314, 73.3415, 73.2276, 73.3405, 73.349, 73.2886, 73.2335, 73.4132],
    'bahdanau': [65.3132, 69.1239, 70.4478, 71.3338, 71.8727, 72.0155, 72.2401, 72.2085, 72.3561, 72.478, 72.6251, 72.2454, 72.3828, 72.617, 72.6475, 72.6277, 72.6614, 72.7389, 72.5882, 72.586],
    'luong': [60.2062, 65.1724, 66.143, 67.229, 67.8514, 68.0839, 68.0818, 68.0348, 68.0738, 67.7733, 67.2584, 67.5856, 66.9376, 66.3879, 65.534, 59.8177, 56.7181, 49.8543, 48.2615, 49.8142]
}
perplexity_valid = {
    'average': [9.1885, 5.61605, 5.04612, 4.97985, 4.90823, 4.97965, 5.04808, 5.09918, 5.12827, 5.24278, 5.22313, 5.41299, 5.29122, 5.37695, 5.43936, 5.56945, 5.6421, 5.55467, 5.5643, 5.62409],
    'multihead': [8.20698, 5.27924, 4.83964, 4.77671, 4.7336, 4.81926, 4.84393, 4.97185, 4.98439, 5.07458, 5.09145, 5.17839, 5.23533, 5.16977, 5.29658, 5.29402, 5.39398, 5.45105, 5.39525, 5.42619],
    'bahdanau': [7.15209, 5.37946, 4.94401, 4.90152, 4.72036, 4.79151, 4.84492, 4.93624, 4.94043, 5.02152, 5.05663, 5.09268, 5.16747, 5.20573, 5.24351, 5.21857, 5.30223, 5.31264, 5.36579, 5.36341],
    'luong': [10.1293, 6.62012, 6.20155, 5.84681, 5.71056, 5.68578, 5.7712, 5.87274, 5.87029, 5.98709, 6.30015, 6.21318, 6.42209, 6.64848, 6.97005, 9.64761, 11.7395, 18.5598, 20.4967, 18.209]
}
perplexity = {
    'average': [68.14, 10.39, 5.45, 3.89, 3.10, 2.69, 2.40, 2.20, 2.06, 1.96, 1.88, 1.81, 1.75, 1.70, 1.66, 1.63, 1.60, 1.58, 1.55, 1.53, 1.51, 1.49, 1.48, 1.46, 1.45, 1.44, 1.42, 1.41, 1.40, 1.39, 1.39, 1.38, 1.37, 1.36, 1.36, 1.35, 1.35, 1.34, 1.33, 1.33],
    'multihead': [64.01, 9.19, 5.06, 3.69, 3.00, 2.60, 2.33, 2.15, 2.01, 1.93, 1.85, 1.78, 1.73, 1.68, 1.65, 1.61, 1.58, 1.56, 1.53, 1.51, 1.49, 1.48, 1.46, 1.44, 1.43, 1.42, 1.41, 1.40, 1.39, 1.38, 1.37, 1.36, 1.36, 1.35, 1.34, 1.34, 1.33, 1.32, 1.32, 1.31],
    'bahdanau': [30.93, 7.92, 4.84, 3.79, 3.12, 2.73, 2.48, 2.29, 2.18, 2.09, 2.02, 1.96, 1.91, 1.87, 1.83, 1.80, 1.77, 1.75, 1.72, 1.71, 1.68, 1.67, 1.65, 1.64, 1.63, 1.62, 1.61, 1.59, 1.59, 1.58, 1.57, 1.57, 1.56, 1.55, 1.54, 1.54, 1.53, 1.52, 1.52, 1.52],
    'luong': [38.72, 11.27, 6.77, 5.28, 4.36, 3.94, 3.57, 3.38, 3.10, 2.99, 2.88, 2.78, 2.78, 2.86, 2.66, 2.58, 2.73, 2.65, 2.61, 2.99, 2.63, 2.61, 2.77, 2.61, 2.63, 2.72, 2.85, 2.97, 2.98, 3.22, 3.94, 5.74, 6.97, 9.18, 10.73, 12.73, 13.08, 14.62, 12.63, 11.45],
}
elapsed_time = {
    'averafge': 6824,
    'multihead': 7154,
    'bahdanau': 5221,
    'luong': 4731,
}
transformer_attn_fn = ['average', 'multihead']
rnn_attn_fn = ['bahdanau', 'luong']
all_attn_fn = transformer_attn_fn + rnn_attn_fn
colors = ['#d81159','#8f2d56','#218380','#73d2de','#fbb13c']

def createDf(data):
    df = pd.DataFrame(data)
    df['steps'] = list(range(1000,41000,1000*int(40/df.shape[0])))
    return df

def drawPplAndLossGraph(ppl_df, loss_df, attn_fn, title):
    plt.clf()
    color_it = cycle(colors)
    for fn in attn_fn:
        plt.plot(ppl_df['steps'][1:], ppl_df[str(fn)][1:], label = str(fn) + ' ppl', color=next(color_it))
        plt.plot(loss_df['steps'], loss_df[str(fn)], label = str(fn) + ' loss', linestyle = '--', color=next(color_it))
    plt.title(title)
    plt.xlabel('TrainingSteps')
    plt.ylabel('CrossEntropyLoss/Perplexity')
    plt.legend()
    plt.savefig(str(title) + '.png')

def drawAcuracyGraph(test_df, val_df, attn_fn, title):
    plt.clf()
    color_it = cycle(colors)
    for fn in attn_fn:
        plt.plot(val_df['steps'], val_df[str(fn)], label = str(fn) + ' valid accuracy', color=next(color_it))
        plt.plot(test_df['steps'], test_df[str(fn)], label = str(fn) + ' train accuracy', linestyle = '--', color=next(color_it))
    plt.title(title)
    plt.xlabel('TrainingSteps')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(str(title) + '.png')

def drawPerplexityGraph(test_df, val_df, attn_fn, title):
    plt.clf()
    color_it = cycle(colors)
    for fn in attn_fn:
        plt.plot(val_df['steps'], val_df[str(fn)], label = str(fn) + ' valid perplexity', color=next(color_it))
        plt.plot(test_df['steps'], test_df[str(fn)], label = str(fn) + ' train perplexity', linestyle = '--', color=next(color_it))
    plt.title(title)
    plt.xlabel('TrainingSteps')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.savefig(str(title) + '.png')

def drawBarGraph(src_lng, trg_lng):
    lines = open('data/%s-%s.txt' % (src_lng, trg_lng), encoding='utf-8').read().strip().split('\n')
    def def_value():
        return 0
    src_len = defaultdict(def_value)
    trg_len = defaultdict(def_value)
    for line in lines:
        line_split = line.split('\t')
        src_word_counter = len(line_split[0].split(' '))
        trg_word_counter = len(line_split[1].split(' '))
        src_len[src_word_counter] = src_len[src_word_counter] + 1
        trg_len[trg_word_counter] = trg_len[trg_word_counter] + 1

    x = list(src_len.keys())
    y = list(src_len.values())
    plt.bar(x,y)
    plt.xlabel("Sentence length(word)")
    plt.ylabel("Counter")
    plt.savefig("Test.png")
    

def main():
    ppl = createDf(perplexity)
    celoss = createDf(loss)
    acc = createDf(accuracy_test)
    acc_val = createDf(accuracy_valid)
    ppl_val = createDf(perplexity_valid)
    
    drawPplAndLossGraph(ppl, celoss, transformer_attn_fn, "Transformer: Perplexity and Loss over Training Steps")
    drawAcuracyGraph(acc, acc_val, transformer_attn_fn, "Transformer: Accuracy over Training Steps")
    drawPerplexityGraph(ppl, ppl_val, transformer_attn_fn, "Transformer: Perplexity over Training Steps")
    
    drawPplAndLossGraph(ppl, celoss, rnn_attn_fn, "RNN: Perplexity and Loss over Training Steps")
    drawAcuracyGraph(acc, acc_val, rnn_attn_fn, "RNN: Accuracy over Training Steps")
    drawPerplexityGraph(ppl, ppl_val, rnn_attn_fn, "RNN: Perplexity over Training Steps")


if __name__ == "__main__":
    drawBarGraph('en', 'deu')
