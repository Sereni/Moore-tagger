__author__ = 'Sereni'
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from sklearn import grid_search
import csv
import numpy


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