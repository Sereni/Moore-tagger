__author__ = 'Sereni'
import os
import xml.etree.ElementTree as ET
import features as ft
import csv

# this module extracts features for training
# uses disambiguated RNC format
# use the API to parse plain text with a trained model
# todo write that API towards the end


class Corpus():

    def __init__(self):
        self.raw_tokens = set([])
        self.tokens = set([])

    def load_file(self, path):
        """
        Open RNC XML and get all unique tokens
        """
        tree = ET.parse(path)
        for elem in tree.iter('w'):

            # find node text. using this instead of elem.text, because the text is after nested elements
            word = self.normalize(''.join(elem.itertext()))
            if word:
                self.raw_tokens.add(word)

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
        """
        for word in self.raw_tokens:
            self.tokens.add(Token(word))

    def to_csv(self):
        """
        Write featurized tokens to csv file
        """
        HEADER = ('token', 'token_lower', 'capital', 'digit', 'hyphen',
                  'prefix1', 'prefix2', 'prefix3', 'prefix4',
                  'suffix1', 'suffix2', 'suffix3', 'suffix4',
                  'shape1', 'shape2')
        with open('feature_matrix.csv', 'w') as out:
            writer = csv.writer(out, delimiter=';', quotechar='"')
            writer.writerow(HEADER)

            for token in self.tokens:
                row = (
                    token.word, token.word_lower, str(int(token.capital)), str(int(token.digit)), str(int(token.hyphen)),
                    token.prefix1, token.prefix2, token.prefix3, token.prefix4,
                    token.suffix1, token.suffix2, token.suffix3, token.suffix4,
                    token.shape1, token.shape2
                       )
                writer.writerow(row)


class Token():
    def __init__(self, word):
        """
        Initalize all the features for a given token
        """

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


def test():
    corpus = Corpus()
    corpus.load_dir(os.path.join(os.getcwd(), "test_corpus"))
    corpus.get_features()
    corpus.to_csv()

if __name__ == '__main__':
    test()