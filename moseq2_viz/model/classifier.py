import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from moseq2_viz.model.util import get_Xy_values
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit

def _classify(features, labels, classifier, train_ind, test_ind):
    '''
    Trains the inputted classifier object on the provided training data, and then tests the model
     on the provided test data. Returning predicted labels and the model performance score.

    Parameters
    ----------
    features (2D numpy array): syllable statistics for each session.
    labels (1D list): group names corresponding to each index in the features array.
    classifier (sklearn OneVsRestClassifier instance): classifier to train and test.
    train_ind (1d list): indices of training data within features and labels
    test_ind (1d list): indices of testing data within features and labels

    Returns
    -------
    classifier (sklearn OneVsRestClassifier instance): trained classifier
    labels (1d list): 1d list of true label names for the predicted test data
    pred_labels (1d list): 1d list of model predicted test labels.
    score (float): percentage value indicating model accuracy.
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
    Returns the initialized model object with user defined tuning parameters.

    Parameters
    ----------
    model_type (str): type of model to initialize; ['logistic_regression', 'svc', 'linearsvc', 'rf']
    C (float): model regularization value.
    penalty (str): model penalty type; ['l1', 'l2']

    Returns
    -------
    model (sklearn model Instance): initialized model.
    model_hyparams (dict): dict of model hyperparameters.
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
    Computes the ROC_AUC score based on the model's prediction performance.

    Parameters
    ----------
    ys (1d list): list of all unique label values.
    true_labels (1d list): list of true label values.
    scores (list): list of scores for each model in the statrified classification
    calc_auc (bool): indicates whether to compute the AUCs

    Returns
    -------
    aucs (1d list): list of ROC-AUC score values for each prediction
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
                        n_cpu=-1, n_fold=10, calc_auc=True, test_size=0.2, whiten=False, random_state=0):
    '''
    Trains and tests multiple OneVsRest Classifiers using StratifiedShuffleSplit to obtain differently arranged
     splits of the data to train the models on.

    Parameters
    ----------
    features (2D numpy array): syllable statistics for each session.
    labels (1D list): group names corresponding to each index in the features array.
    labels_for_stratification (1D list): group names corresponding to each index in the features array.
    model_type (str): type of model to initialize; ['logistic_regression', 'svc', 'linearsvc', 'rf']
    C (float): model regularization value.
    penalty (str): model penalty type; ['l1', 'l2']
    n_cpu (int): number of cpus to dedicate to model training and testing
    n_fold (int): number of splits to stratify the data by.
    calc_auc (bool): indicates whether to compute an AUC score for the model
    test_size (float): indicates the proportion to split the train-test data. Values are between (0, 1).
    whiten (bool): indicates whether to whiten the features prior to model training.

    Returns
    -------
    classifiers (1d list): list of trained classifiers.
    true_labels (2d list): list of true label value lists for each corresponding classifier
    pred_labels (2d list): list of predicted label value lists for each corresponding classifier
    scores (1d list): list of classifier performance scores.
    aucs (1d list): list of classifier ROC-AUC scores.
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
                                                        random_state=random_state).split(features, labels_for_stratification))

    classifiers, true_labels, pred_labels, scores = zip(*out)

    aucs = compute_aucs(ys, true_labels, scores, calc_auc)

    return classifiers, true_labels, pred_labels, scores, aucs

def run_classifier(mean_df, stat='usage', output_file='confusion_matrix.pdf', normalize=True, random_state=0):
    '''
    Main access point to train and test stratified classifiers on MoSeq data, accessible from the
     MoSeq2-Results-Extension Notebook.
     Function will also generate a confusion matrix displaying the trained classifier results.

    Parameters
    ----------
    mean_df (pd.DataFrame): mean syllable statistics dataframe.
    stat (str): statistic to train classifiers on.
    output_file (str): path to file to save confusion matrix figure.
    normalize (bool): indicates whether to normalize the confusion matrix color scheme.

    Returns
    -------
    return_dict (dict): dict containing all classifier results.
    fig (matplotlib figure): figure containing plotted confusion matrix.
    ax (matplonlib axes): axes instance for plotted figure.
    '''

    syllable_df = mean_df.groupby(['syllable', 'uuid', 'group'], as_index=False).mean()

    unique_groups = sorted(syllable_df.group.unique())

    X, y, mapping, rev_mapping = get_Xy_values(syllable_df, unique_groups, stat=stat)

    classifiers, true_labels, pred_labels, scores, aucs = classify_stratified(X, y, y,
                                                                              whiten=True, random_state=random_state)

    fig, ax = compute_group_confusion_matrix(true_labels,
                                             pred_labels,
                                             mapping,
                                             output_file=output_file,
                                             normalize=normalize)

    return_dict = {
        'classifiers': classifiers,
        'true_labels': true_labels,
        'pred_labels': pred_labels,
        'scores': scores,
        'aucs': aucs
    }

    return return_dict, fig, ax

def kl_divergence(a, b):
    '''

    Parameters
    ----------
    a (list):
    b (list):

    Returns
    -------
    kl-distance (float):
    '''

    return sum(a[i] * np.nan_to_num(np.log(a[i] / b[i])) for i in range(len(a)))

def compute_kl_distance_matrix(syllable_df, unique_groups, stat='usage'):
    '''

    Parameters
    ----------
    syllable_df (pd.DataFrame):
    unique_groups (str):
    stat (str):

    Returns
    -------

    '''

    dists = []
    for g in unique_groups:
        g_dists = []
        for h in unique_groups:
            a = syllable_df[syllable_df['group'] == g].groupby('syllable')[stat].mean().fillna(0).to_numpy()
            b = syllable_df[syllable_df['group'] == h].groupby('syllable')[stat].mean().fillna(0).to_numpy()
            g_dists.append(kl_divergence(a, b))

        dists.append(np.array(g_dists))

    return np.array(dists)

def compute_and_plot_kl_group_distance_matrix(mean_df, stat='usage', normalize=True, output_file='group_kl_distance.pdf'):
    '''

    Parameters
    ----------
    mean_df (pd.DataFrame): mean syllable statistics dataframe.
    stat (str): statistic to train classifiers on.
    normalize (bool): indicates whether to normalize the confusion matrix color scheme.
    output_file (str): path to file to save confusion matrix figure.

    Returns
    -------
    fig (matplotlib figure): figure containing plotted confusion matrix.
    ax (matplonlib axes): axes instance for plotted figure.
    '''

    # Get mean syllable mean DataFrame and unique group name
    syllable_df = mean_df.groupby(['syllable', 'uuid', 'group'], as_index=False).mean()
    unique_groups = sorted(syllable_df.group.unique())

    # Compute KL distance matrix
    cm_dists = compute_kl_distance_matrix(syllable_df, unique_groups, stat=stat)

    # Normalize distances
    if normalize:
        cm_dists = cm_dists.astype('float') / cm_dists.sum(axis=1)[:, np.newaxis]

    # plot confusion matrix
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))

    im = ax.imshow(cm_dists, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)

    # add annotation
    ax.set(xticks=np.arange(cm_dists.shape[1]),
           yticks=np.arange(cm_dists.shape[0]),
           xticklabels=unique_groups, yticklabels=unique_groups,
           title='Behavioral Similarity Between Cohorts (KL Distance)',
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Save figure
    fig.tight_layout()
    fig.savefig(output_file, bbox_inches='tight', format='pdf')

    return fig, ax

def compute_group_confusion_matrix(true_labels,
                                   pred_labels,
                                   mapping,
                                   output_file='confusion_matrix.pdf',
                                   title='Group Confusion Matrix',
                                   normalize=True):
    '''
    Computes and graphs a confusion matrix based on the true and predicted label results from a trained
     classifier instance.

    Parameters
    ----------
    true_labels (2d list): list of true labels for each classifier's stratified set of true values.
    pred_labels (2d list): list of predicted labels from each classifier's stratified set of predicted values.
    mapping (dict): dict containing group name to integer mappings keep track of label names.
    output_file (str): path to saved outputted figure in.
    title (str): figure title.
    normalize (bool): indicates whether to normalize the confusion matrix color scheme.

    Returns
    -------
    fig (matplotlib figure): figure containing plotted confusion matrix.
    ax (matplonlib axes): axes instance for plotted figure.
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
