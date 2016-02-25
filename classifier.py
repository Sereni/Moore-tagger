__author__ = 'Sereni'
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
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

    with open(path) as f:
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

# todo add Elya's word clusters
if __name__ == '__main__':

    # import data
    data, target = import_as_dict('/Users/Sereni/PycharmProjects/Moore-tagger/feature_matrix.csv')

    # todo early stopping
    # initialize classifier
    clf = SGDClassifier(loss='hinge',              # hinge loss objective
                        shuffle=True,              # shuffle samples before learning (required)
                        learning_rate='constant',  # constant learning rate
                        eta0=2**(-8),              # of 2^-8
                        average=True               # I hope that's the setting for averaged perceptron...
                        )
    # split data into train and test subsets
    data_train, data_test, target_train, target_test = train_test_split(data, target)
    clf.fit(data_train, target_train)
    clf.score(data_test, target_test)  # fixme scoring does not work

    joblib.dump(clf, 'model.pkl')
    # todo grid search and save best model
    # todo progress updates or something