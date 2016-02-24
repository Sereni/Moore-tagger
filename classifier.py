__author__ = 'Sereni'

# using SVM hinge loss objective w/ gradient descent (not a simple linear SVC)
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
import csv
import numpy

# todo run through with your features
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



# todo add Elya's word clusters
if __name__ == '__main__':

    # import data
    data, target = import_csv('/Users/Sereni/PycharmProjects/Moore-tagger/feature_matrix.csv')
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
    clf.fit(data_train, target_train)  # fixme ValueError: could not convert string to float: 'асфальте'
    clf.score(data_test, target_test)

    joblib.dump(clf, 'model.pkl')
    # todo grid search and save best model
    # todo progress updates or something