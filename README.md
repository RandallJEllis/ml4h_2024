# Machine Learning Experiments
This project contains a set of Python scripts for running machine learning experiments, particularly focused on predicting outcomes using proteomics and demographic data.


## Files
1. `ml_experiments.py`: Main script for running the experiments.
2. `bootstrap.py`: Contains functions for bootstrap analysis.
3. `ml_utils.py`: Utility functions for machine learning tasks.


## Dependencies
- pandas
- numpy
- scikit-learn
- flaml
- argparse

## Main Script: ml_experiments.py
The main script handles the following tasks:

1. Loading and preprocessing data
2. Splitting data into train and test sets
3. Running AutoML for model selection
4. Performing bootstrap analysis on training and test sets

### Usage

```
python ml_experiments.py --task_id <task_id> --outcome <outcome> --output_path <output_path> --data_path <data_path>
```


### Key Functions
def load_datasets(output_path):
    X = pd.read_parquet(f'{output_path}/X.parquet')
    y = np.load(f'{output_path}/y.npy')
    return X, y


def load_proteomics(data_path):
    df = pd.read_parquet(data_path + 'proteomics/proteomics.parquet')

    # only keep proteins from Instance 0
    cols_remove = []
    for c in df.columns:
        if '-1' in c:
            cols_remove.append(c)
        elif '-2' in c:
            cols_remove.append(c)
        elif '-3' in c:
            cols_remove.append(c)
    df = df.drop(columns=cols_remove)
    df = df.dropna(subset=df.columns[1:], how='all')

    return df


These functions load datasets, proteomics data, and demographic information.


## Bootstrap Analysis: bootstrap.py
This script contains the `run_bootstrap` function, which performs bootstrap analysis on the model predictions.


## Utility Functions: ml_utils.py

This file contains various utility functions for machine learning tasks, including:

1. Encoding categorical variables
2. Threshold selection
3. Calculating performance metrics
   
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


## Workflow
1. Data is loaded and preprocessed.
2. Features are selected based on the specified outcome.
3. Data is split into training and test sets.
4. AutoML is used to find the best model.
5. Bootstrap analysis is performed on both training and test sets.
6. Results are saved to specified output paths.