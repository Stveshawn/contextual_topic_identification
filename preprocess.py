import numpy as np
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from language_detector import detect_language

from nltk.stem import WordNetLemmatizer
import pkg_resources
from symspellpy import SymSpell, Verbosity


# language detection
def f_lan(s):
    """
    :param s: string to be processed
    :return: boolean (s is English)
    """
    return detect_language(s) == 'English'


# lowercase and filtering out the obvious abnormal
def f_lower(s):
    """
    :param s: string to be processed
    :return: lower-cased s
    """
    return s.lower()


def f_base(s):
    """
    :param s: string to be processed
    :return: processed string:
            1. delete &gt, &lt
            2. delete repetition if more than 3
    """
    s = re.sub(r'&gt|&lt', ' ', s)
    s = re.sub(r'([a-z])\1{3,}', r'\1', s)
    return s
# rws = data.review.values
# rws = list(map(lambda w: w.lower(), rws))
# rws = list(map(lambda w: re.sub(r'&gt|&lt', ' ', w), rws))
# rws = list(map(lambda w: re.sub(r'([a-z])\1{3,}', r'\1', w), rws))



# # filtering out non-english reviews
# q = []
# for _ in rws:
#     if detect_language(_) == 'English':
#         q.append(_)

# filtering out stop words
def f_stopw(s, stop_words):
    """
    filtering out stop words
    """
    return [word for word in s if word not in stop_words]

# filtering out punctuations
def f_punct(s):
    """
    filting out punctuations and numbers
    """
    return [word for word in s if word.isalpha()]

# lemmatization
lemmatizer = WordNetLemmatizer()
def f_lem(w):
    """
    lemmatization
    """
    return [lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word, pos='v'), pos='a'), pos='n') for word in w]

# other normalization
# rly -> really
# aggro -> aggressive


# typo correction using symspellpy
sym_spell = SymSpell(max_dictionary_edit_distance=4, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
# term_index is the column of the term and count_index is the
# column of the term frequency
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)


# typo correction
def f_typo(s):
    """
    :param s: string to be processed (list of words)
    :return: s with typo fixed
    """
    s_typo_fixed = []
    for word in s:
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=3)
        if suggestions:
            s_typo_fixed.append(suggestions[0].term)
        else:
            pass
            # do word segmentation
            # w_seg = sym_spell.word_segmentation(phrase=word)
            # s_typo_fixed.extend(w_seg.corrected_string.split())
    return s_typo_fixed




#### preprocessing
stop_words = set(stopwords.words('english'))
stop_words = stop_words.union(set(['cant', 'cannot', 'can\'t', 'wont', 'won\'t']))


def preprocess(s):
    """
    :param s: string to be processed
    :return: list of processed words
    """
    if not f_lan(s):
        return
    s = f_lower(s)
    s = f_base(s)
    s = word_tokenize(s)
    s = f_punct(s)
    s = f_lem(s)
    s = f_stopw(s, stop_words)
    s = f_typo(s)
    return s

