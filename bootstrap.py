import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, roc_auc_score, \
     precision_recall_fscore_support

from sklearn.metrics import precision_recall_curve, average_precision_score, \
     accuracy_score, balanced_accuracy_score, roc_curve

from datetime import datetime

import pickle

import ml_utils



def run_bootstrap(fpath, fname, true_labels, proba_predict, n_bootstrap,
                  beta=1, proteins=None, threshold=None, youden=False,
                  return_results=False):
    # store results
    bs_l = []
    true_labels_l = []
    probas_l = []
    tn_l = []
    fp_l = []
    fn_l = []
    tp_l = []
    auroc_l = []
    ap_l = []
    best_score_l = []
    best_threshold_l = []
    accuracy_l = []
    balanced_accuracy_l = []
    prec_n = []
    prec_p = []
    rec_n = []
    rec_p = []
    fscore_n = []
    fscore_p = []

    bootstraps = np.array([np.random.choice(range(len(true_labels)),
                           len(true_labels), replace=True)
                           for _ in range(n_bootstrap)])
    for j, bs_index in enumerate(bootstraps):
        if j % 1000 == 0:
            print(j)
        yhat = proba_predict[bs_index]
        y_bs = true_labels[bs_index]

        auroc = roc_auc_score(y_bs,
                              yhat)
        ap = average_precision_score(y_bs, yhat)

        if threshold is None:
            best_score, best_threshold = pick_threshold(true_labels,
                                                        proba_predict,
                                                        beta=beta,
                                                        youden=youden)
            test_pred = (yhat > best_threshold).astype(int)
            best_threshold_l.append(best_threshold)
        else:
            best_score = np.nan
            test_pred = (yhat > threshold).astype(int)
            best_threshold_l.append(threshold)

        tn, fp, fn, tp = confusion_matrix(y_bs,
                                          test_pred).ravel()
        acc = accuracy_score(y_bs, test_pred)
        bal_acc = balanced_accuracy_score(y_bs,
                                          test_pred)
        prfs = precision_recall_fscore_support(y_bs,
                                               test_pred,
                                               beta=beta)
        bs_l.append(j)
        true_labels_l.append(y_bs)
        probas_l.append(yhat)
        auroc_l.append(auroc)
        ap_l.append(ap)
        best_score_l.append(best_score)
        tn_l.append(tn)
        fp_l.append(fp)
        fn_l.append(fn)
        tp_l.append(tp)
        accuracy_l.append(acc)
        balanced_accuracy_l.append(bal_acc)
        prec_n.append(prfs[0][0])
        prec_p.append(prfs[0][1])
        rec_n.append(prfs[1][0])
        rec_p.append(prfs[1][1])
        fscore_n.append(prfs[2][0])
        fscore_p.append(prfs[2][1])

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Analyses complete. Current Time =", current_time)
    data = {'bootstrap': bs_l,
            'TN': tn_l, 'FP': fp_l,
            'FN': fn_l, 'TP': tp_l, 'auroc': auroc_l, 'avg_prec': ap_l,
            'best_thresh': best_threshold_l, f'best_f{beta}': best_score_l,
            'accuracy': accuracy_l,
            'balanced_acc': balanced_accuracy_l, 'prec_neg': prec_n,
            'prec_pos': prec_p, 'rec_neg': rec_n, 'rec_pos': rec_p,
            f'f{beta}_neg': fscore_n, f'f{beta}_pos': fscore_p}
    results = pd.DataFrame(data)
    print('dataframe made')
    if proteins is not None:
        results['proteins'] = [proteins] * len(results)
        print('added proteins to dataframe')
    results.to_parquet(f'{fpath}/{fname}_bootstrap_results.parquet')

    # Save all true labels
    file_path = f'{fpath}/{fname}_all_true_labels.pkl'
    with open(file_path, 'wb') as file:
        # Serialize and write the nested list to the file
        pickle.dump(true_labels_l, file)

    # Save all predicted probabilities
    file_path = f'{fpath}/{fname}_all_probas.pkl'
    with open(file_path, 'wb') as file:
        # Serialize and write the nested list to the file
        pickle.dump(probas_l, file)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Results saved. All done. Current Time =", current_time)

    if return_results is True:
        return results
