import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, \
     precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve, average_precision_score, \
     accuracy_score, balanced_accuracy_score, roc_curve, auc
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import os
from utils import save_pickle
from sklearn.preprocessing import OneHotEncoder

def encode_categorical_vars(df, catcols):
    # encode sex, ethnicity, APOEe4 alleles, education qualifications
    enc = OneHotEncoder(drop='if_binary')
    enc.fit(df.loc[:, catcols])
    categ_enc = pd.DataFrame(enc.transform(df.loc[:, catcols]).toarray(),
                            columns=enc.get_feature_names_out(catcols))
    return categ_enc

def pick_threshold(y_true, y_probas, youden=False, beta=1):
    scores = []

    if youden is True:
        # calculate roc curve
        fpr, tpr, thresholds = roc_curve(y_true, y_probas)
        
        for i, t in enumerate(thresholds):
            # youden index = sensitivity + specificity - 1
            # AKA sensitivity + (1 - FPR) - 1 (NOTE: (1-FPR) = TNR)
            # AKA recall_1 + recall_0 - 1
            youdens_j = tpr[i] + (1 - fpr[i]) - 1
            scores.append(youdens_j)

    else:
        # calculate pr-curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_probas)

        # convert to f score
        for i, t in enumerate(thresholds):
            fscore = ((1 + beta**2) * precision[i] * recall[i]) / \
                        ((beta**2 * precision[i]) + recall[i])
            scores.append(fscore)

    ix = np.nanargmax(scores)
    best_threshold = thresholds[ix]

    return best_threshold

def calc_results(y_true, y_probas, youden=False, beta=1, threshold=None):
    auroc = roc_auc_score(y_true, y_probas)
    ap = average_precision_score(y_true, y_probas)

    return_threshold = False
    if threshold is None:
        threshold = pick_threshold(y_true, y_probas, youden, beta)
        return_threshold = True

    test_pred = (y_probas >= threshold).astype(int)
            
    tn, fp, fn, tp = confusion_matrix(y_true, test_pred).ravel()
    acc = accuracy_score(y_true, test_pred)
    bal_acc = balanced_accuracy_score(y_true,
                                      test_pred)
    prfs = precision_recall_fscore_support(y_true,
                                           test_pred, beta=beta)
    # print(f'AUROC: {auroc}, AP: {ap}, Fscore: {best_fscore}, Accuracy: {acc}, Bal. Acc.: {bal_acc}, Best threshold: {best_threshold}')
    print(f'AUROC: {np.round(auroc, 4)}, AP: {np.round(ap, 4)}, \nAccuracy: {np.round(acc, 4)}, Bal. Acc.: {np.round(bal_acc, 4)}, \nBest threshold: {np.round(threshold, 4)}')
    print(f'Precision/Recall/Fscore: {prfs}')
    print('\n')
    res =  pd.Series(data=[auroc, ap, threshold, tp, tn, fp, fn, acc, bal_acc,
                           prfs[0][0], prfs[0][1], prfs[1][0], prfs[1][1],
                            prfs[2][0], prfs[2][1]], 
                            index=['auroc', 'avg_prec', 'threshold', 'TP', 'TN', 'FP', 'FN',
                                   'accuracy', 'bal_acc', 'prec_n', 'prec_p', 'recall_n', 'recall_p',
                                    f'f{beta}_n', f'f{beta}_p'])
    if return_threshold == True:
        return res, threshold
    else:
        return res


def save_labels_probas(filepath, train_labels, train_probas, test_labels, test_probas):
    save_pickle(f'{filepath}/train_true_labels.pkl', train_labels)
    save_pickle(f'{filepath}/train_probas.pkl', train_probas)
    save_pickle(f'{filepath}/test_true_labels.pkl', test_labels)
    save_pickle(f'{filepath}/test_probas.pkl', test_probas)

