__author__ = 'Sereni'

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from sklearn import grid_search
import csv
import numpy
from scipy import interp


def import_csv(path):
    """
    Import feature data from a given csv file
    :param path: path to CSV file containing tokens and features
    :return features and target tags as numpy arrays
    """
    with open(path) as f:
        reader = csv.reader(f, delimiter=';')
        next(reader, None)  # skip header

        data = []
        target = []
        for row in reader:
            data.append(row[:-1])
            target.append(row[-1])

        # convert to numpy arrays
        data = numpy.array(data)
        target = numpy.array(target)
    return data, target


def import_as_dict(path):
    """
    Import feature data from a given csv file
    :param path: path to CSV file containing tokens and features
    :return features and target tags as sparse matrices
    Should deal with categorical input.
    """

    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        header = next(reader, None)

        data = []
        target = []

        # read things from csv
        for row in reader:
            data.append(dict(zip(header[:-1], row[:-1])))  # make a dict of feature : value
            target.append(row[-1])

        from sklearn.feature_extraction import DictVectorizer
        vec = DictVectorizer()

        # convert categorical features to floats
        data_matrix = vec.fit_transform(data)

        # convert targets to numpy array as strings
        target_matrix = numpy.array(target)

        # save converter to use in prediction
        joblib.dump(vec, 'feature_transformer.pkl')

    return data_matrix, target_matrix


def greedy():
    # Set the parameters by cross-validation
    tuned_parameters = [{'loss': ['hinge', 'log'], 'shuffle': [True],
                         'learning_rate': ['constant'], 'eta0': [2**(-8)], 'average': [True, False],
                         'penalty': ['l1', 'l2', 'elasticnet'],
                         'alpha': [0.001, 0.0001, 0.00001, 0.000001]}]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = grid_search.GridSearchCV(SGDClassifier(), tuned_parameters, cv=5,
                           scoring='%s_weighted' % score, verbose=2)
        clf.fit(data_train, target_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = target_test, clf.predict(data_test)
        print(classification_report(y_true, y_pred))
        best = clf.best_estimator_
        print(clf.best_score_)

        joblib.dump(best, 'model2.pkl')
        print()


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    # import data
    print("Import data")
    data, target = import_as_dict('feature_matrix_clusters.csv')
    # todo early stopping

    # split data into train and test subsets
    print("Split data")
    data_train, data_test, target_train, target_test = train_test_split(data, target)
    y = label_binarize(target, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    n_classes = y.shape[1]
    # greedy()

    clf = SGDClassifier(penalty='elasticnet', eta0=0.00390625, learning_rate='constant', alpha=1e-06, loss='hinge')
    y_score = clf.fit(data_train, target_train).decision_function(data_test)
    print(y_score)

    print('Going ROCs!')
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(target_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(target_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    ##############################################################################
    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.plot(fpr[2], tpr[2], label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


    ##############################################################################
    # Plot ROC curves for the multiclass problem

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = numpy.unique(numpy.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = numpy.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             linewidth=2)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             linewidth=2)

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()