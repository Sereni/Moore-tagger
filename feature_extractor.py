__author__ = 'Sereni'
import os
import xml.etree.ElementTree as ET
import features as ft
import csv
import numpy


# this module extracts features for training
# uses disambiguated RNC format
# use the API to parse plain text with a trained model
# todo write that API towards the end


class Corpus():
    def __init__(self):
        self.raw_tokens = set([])
        self.tokens = set([])
        self.feature_array = None

    def load_file(self, path):
        """
        Open RNC XML and get all unique tokens
        """
        tree = ET.parse(path)
        for elem in tree.iter('w'):

            # find node text. using this instead of elem.text, because the text is after nested elements
            word = self.normalize(''.join(elem.itertext()))

            # get POS tag
            for item in elem.iter('ana'):
                tag = item.get("gr").split('=')[0].split(',')[0]
                break

            if word:
                self.raw_tokens.add((word, tag))  # todo а что делать с омонимией? писать, не писать?

    def load_dir(self, path):
        """
        Traverse a given directory and add all text files
        :param path: path to corpus folder
        """
        for root, dirs, files in os.walk(path):
            for name in files:
                if name.endswith('ml'):  # todo open all files, but throw warnings if they are not corpus files
                    self.load_file(os.path.join(root, name))

    def normalize(self, word):
        """
        Replace all digits with 0's
        Any other normalization will go here if we need it
        """
        try:
            for d in '123456789':
                word = word.replace(d, '0')
            word = word.replace('`', '')  # remove stress
        except AttributeError:
            pass
        return word

    def get_features(self):
        """
        Run all tokens through feature extraction
        and store in a separate set
        Token is a tuple of (word, POS)
        """
        for item in self.raw_tokens:
            self.tokens.add(Token(item))

    def make_array(self):

        for token in self.tokens:

            # create row from token features
            row = (
                    token.word, token.word_lower, str(int(token.capital)), str(int(token.digit)),
                    str(int(token.hyphen)),
                    token.prefix1, token.prefix2, token.prefix3, token.prefix4,
                    token.suffix1, token.suffix2, token.suffix3, token.suffix4,
                    token.shape1, token.shape2, token.pos
                )

            if self.feature_array is None:
                self.feature_array = numpy.array(row)

            else:
                # add new row to array
                self.feature_array = numpy.vstack([self.feature_array, row])

    def to_array(self):
        """
        Dump feature array to file (create array if needed)
        """
        if self.feature_array is None:
            self.make_array()
        self.feature_array.dump('feature_array.dat')

    def to_csv(self):
        """
        Write featurized tokens to csv file
        """
        HEADER = ('token', 'token_lower', 'capital', 'digit', 'hyphen',
                  'prefix1', 'prefix2', 'prefix3', 'prefix4',
                  'suffix1', 'suffix2', 'suffix3', 'suffix4',
                  'shape1', 'shape2', 'POS')
        with open('feature_matrix.csv', 'w') as out:
            writer = csv.writer(out, delimiter=';', quotechar='"')
            writer.writerow(HEADER)

            for token in self.tokens:
                row = (
                    token.word, token.word_lower, str(int(token.capital)), str(int(token.digit)),
                    str(int(token.hyphen)),
                    token.prefix1, token.prefix2, token.prefix3, token.prefix4,
                    token.suffix1, token.suffix2, token.suffix3, token.suffix4,
                    token.shape1, token.shape2, token.pos
                )
                writer.writerow(row)


class Token():
    def __init__(self, item):
        """
        Initalize all the features for a given token
        """

        word = item[0]
        self.capital = ft.contains_capital(word)
        self.digit = ft.contains_digit(word)
        self.hyphen = ft.contains_hyphen(word)

        self.prefix1 = ft.prefix(word, 1)
        self.prefix2 = ft.prefix(word, 2)
        self.prefix3 = ft.prefix(word, 3)
        self.prefix4 = ft.prefix(word, 4)

        self.suffix1 = ft.suffix(word, 1)
        self.suffix2 = ft.suffix(word, 2)
        self.suffix3 = ft.suffix(word, 3)
        self.suffix4 = ft.suffix(word, 4)

        self.shape1 = ft.shape1(word)
        self.shape2 = ft.shape2(word)
        self.word = word
        self.word_lower = word.lower()

        self.pos = item[1]
        
        self.features = (self.word, self.word_lower, int(self.capital), int(self.digit), int(self.hyphen),
                         self.prefix1, self.prefix2, self.prefix3, self.prefix4,
                         self.suffix1, self.suffix2, self.suffix3, self.suffix4,
                         self.shape1, self.shape2, self.pos)
        self.feature_names = ('token', 'token_lower', 'capital', 'digit', 'hyphen',
                              'prefix1', 'prefix2', 'prefix3', 'prefix4',
                              'suffix1', 'suffix2', 'suffix3', 'suffix4',
                              'shape1', 'shape2', 'POS')
        self.features_dict = dict(zip(self.feature_names[:-1], self.features[:-1]))


def test():
    corpus = Corpus()
    corpus.load_dir(os.path.join(os.getcwd(), "test_corpus"))
    corpus.get_features()
    corpus.to_csv()
    # corpus.to_array() # fixme this is taking VERY long


def run():
    corpus = Corpus()
    corpus.load_dir(os.path.join(os.getcwd(), "full_corpus"))
    corpus.get_features()
    corpus.to_csv()


if __name__ == '__main__':
    # test()
    run()
