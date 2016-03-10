__author__ = 'Sereni'
from sklearn.externals import joblib
from feature_extractor import Token

clf = joblib.load('model2.pkl')
vec = joblib.load('feature_transformer2.pkl')

def pos(word):
    """
    Determine the word's part of speech using our model
    >>> pos('тёплый')
    'A'

    >>> pos('кот')
    'S'

    >>> pos('мяукает')
    'V'

    >>> pos('кросскатегориальность')
    'S'

    >>> pos('шпиль')
    'S'

    >>> pos('стекло')
    'S'

    >>> pos('мой')
    'A-PRO'

    >>> pos('он')
    'S-PRO'
    """
    features = Token((word, None)).features_dict
    vector = vec.transform(features)
    return clf.predict(vector)[0][0]

if __name__ == "__main__":
    # clf = joblib.load('model.pkl')
    # vec = joblib.load('feature_transformer.pkl')

    # demo part
    phrase = input('Введите фразу: ')
    words = phrase.split()
    for word in words:
        print(word, pos(word))

    #import doctest
    #doctest.testmod()