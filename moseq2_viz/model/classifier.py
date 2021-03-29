import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from moseq2_viz.model.embed import get_Xy_values
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit

def _classify(features, labels, classifier, train_ind, test_ind):
    '''

    Parameters
    ----------
    features
    labels
    classifier
    train_ind
    test_ind

    Returns
    -------

    '''

    classifier.fit(features[train_ind], labels[train_ind])
    if hasattr(classifier, "predict_proba"):
        score = classifier.predict_proba(features[test_ind])
    elif hasattr(classifier, "decision_function"):
        score = classifier.decision_function(features[test_ind])
    else:
        raise ValueError(
            "Classifier %s has neither a probability or decision score prediction method" % str(classifier))
    pred_labels = classifier.predict(features[test_ind])
    return classifier, labels[test_ind], pred_labels, score

def select_model(model_type, C, penalty):
    '''

    Parameters
    ----------
    model_type
    C
    penalty

    Returns
    -------

    '''

    if model_type == "logistic_regression":
        model = LogisticRegression
        model_hyparams = dict(penalty=penalty, C=C, class_weight='balanced')
    elif model_type == "svc":
        model = SVC
        model_hyparams = dict(
            kernel='rbf', C=C, class_weight='balanced', decision_function_shape='ovr')
    elif model_type == "linearsvc":
        model = LinearSVC
        model_hyparams = dict(C=C, class_weight='balanced')
    elif model_type == 'rf':
        model = RandomForestClassifier
        model_hyparams = dict(n_estimators=100, class_weight='balanced')
    else:
        raise ValueError(
            "Valid model_type values: ('logistic_regression','svc','linearsvc','rf')")

    return model, model_hyparams

def compute_aucs(ys, true_labels, scores, calc_auc=False):
    '''

    Parameters
    ----------
    ys
    true_labels
    scores
    calc_auc

    Returns
    -------

    '''

    aucs = []

    if calc_auc:
        for i in ys:
            aucs.append([])
            for tl, sc in zip(true_labels, scores):
                aucs[-1].append(roc_auc_score(tl == i, sc[:, i]))

    aucs = np.array(aucs)
    return aucs

def classify_stratified(features, labels, labels_for_stratification,
                        model_type="logistic_regression", C=1.0, penalty='l2',
                        n_cpu=-1, n_fold=10, calc_auc=True, test_size=0.2, whiten=False):
    '''

    Parameters
    ----------
    features
    labels
    labels_for_stratification
    model_type
    C
    penalty
    n_cpu
    n_fold
    calc_auc
    test_size
    whiten

    Returns
    -------

    '''

    ys = np.unique(labels)

    assert len(features) == len(labels) == len(
        labels_for_stratification), "Features, labels and mouse IDs don't match"

    model, model_hyparams = select_model(model_type=model_type, C=C, penalty=penalty)

    if whiten:
        proc_features = PCA(0.99, whiten=True).fit_transform(features)
    else:
        proc_features = features.copy()

    out = Parallel(n_jobs=n_cpu)(delayed(_classify)(proc_features,
                                                    labels,
                                                    OneVsRestClassifier(model(**model_hyparams)),
                                                    train_ind, test_ind) for train_ind, test_ind in
                                 # this is where we would switch from validation to test
                                 StratifiedShuffleSplit(n_splits=n_fold,
                                                        test_size=test_size,
                                                        random_state=0).split(features, labels_for_stratification))

    classifiers, true_labels, pred_labels, scores = zip(*out)

    aucs = compute_aucs(ys, true_labels, scores, calc_auc)

    return classifiers, true_labels, pred_labels, scores, aucs

def run_classifier(mean_df, stat='usage', output_file='confusion_matrix.pdf', normalize=True):
    '''

    Parameters
    ----------
    mean_df
    stat
    output_file
    normalize

    Returns
    -------

    '''

    syllable_df = mean_df.groupby(['syllable', 'uuid', 'group'], as_index=False).mean()

    unique_groups = syllable_df.group.unique()

    X, y, mapping, rev_mapping = get_Xy_values(syllable_df, unique_groups, stat=stat)

    classifiers, true_labels, pred_labels, scores, aucs = classify_stratified(X, y, y, whiten=True)

    fig, ax = compute_group_confusion_matrix(true_labels,
                                             pred_labels,
                                             mapping,
                                             output_file=output_file,
                                             normalize=normalize)

def compute_group_confusion_matrix(true_labels,
                                   pred_labels,
                                   mapping,
                                   output_file='confusion_matrix.pdf',
                                   title='Group Confusion Matrix',
                                   normalize=True):
    '''

    Parameters
    ----------
    true_labels
    pred_labels
    mapping
    output_file
    title
    normalize

    Returns
    -------

    '''

    # stack the results
    stacked_true_y = np.hstack(true_labels)
    stacked_pred_y = np.hstack(pred_labels)

    # compute confusion matrix
    cm = confusion_matrix(stacked_true_y, stacked_pred_y)

    # normalize
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # plot confusion matrix
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))

    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)

    # add annotation
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=list(mapping.keys()), yticklabels=list(mapping.keys()),
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Save figure
    fig.tight_layout()
    fig.savefig(output_file, bbox_inches='tight', format='pdf')

    return fig, ax
