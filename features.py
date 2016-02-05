import re


def contains_capital(word):
    """
    Check whether a word contains a capital letter
    :type word: str
    """
    if word.lower() == word:
        return False

    return True


def contains_digit(word):
    for digit in '0123456789':  # smarter way?
        if digit in word:
            return True
    return False


def contains_hyphen(word):
    for hyphen in '―–‒-—':  # added other hyphens; probably, some of them are dashes
        if hyphen in word:
            return True
    return False


def prefix(word, n):
    """
    Return an n-prefix of a word
    """
    return word.lower()[:n]


def suffix(word, n):
    """
    Return an n-suffix of a word
    """
    return word.lower()[-n:]


def shape1(word):
    transformed = ''
    for char in word:
        if char.islower():
            transformed += 'x'
        elif char.isupper():
            transformed += 'X'
        else:
            transformed += '0'
    return transformed


def shape2(word):
    s = shape1(word)
    s = re.sub('X+', 'X', s)  # performance hit? # why?
    s = re.sub('x+', 'x', s)
    s = re.sub('0+', '0', s)
    return s


def full_word(word):
    return word


def full_word_lower(word):
    return word.lower()

# todo clusters