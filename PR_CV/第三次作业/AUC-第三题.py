# -*- coding: UTF-8 -*- #

from collections import Counter


v = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
label = [1, 2, 1, 1, 2, 1, 2, 2, 1, 2]#[1, 2, 1, 1, 2, 1, 2, 2, 2, 1]

P = [1.0]
R = [0.0]
TPR = [0.0]
FPR = [0.0]
for i in range(1, len(v) + 1):
    pos_count = Counter(label[:i])
    neg_count = Counter(label[i:])
    TP = pos_count.get(1, 0)
    FP = pos_count.get(2, 0)
    FN = neg_count.get(1, 0)
    TN = neg_count.get(2, 0)
    P.append(TP / (TP + FP))
    R.append(TP / (TP + FN))
AUC_PR = [0.5 * (R[i] - R[i - 1]) * (P[i] + P[i - 1]) for i in range(1, len(R))]
AP = [(R[i] - R[i - 1]) * P[i] for i in range(1, len(R))]

print('P:', [*P])
print('R:', [*R])
print('AUC_PR:', [*AUC_PR])
print('AP:', [*AP])