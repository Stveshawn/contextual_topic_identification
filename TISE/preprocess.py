import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
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

# some basic normalization
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

# filtering out stop words
#### preprocessing
stop_words = set(stopwords.words('english'))
stop_words = stop_words.union(set(['cant', 'cannot', 'can\'t', 'wont', 'won\'t']))

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


# filter nouns
def f_noun(w):
    """
    selecting nouns
    """
    return [word for (word, pos) in nltk.pos_tag(w) if pos[:2] == 'NN']


# # typo correction using symspellpy
print("Loading symspell dictionary ...")
sym_spell = SymSpell(max_dictionary_edit_distance=4, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
# term_index is the column of the term and count_index is the
# column of the term frequency
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
print("Loading symspell dictionary done.")

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





def preprocess(rw):
    """
    :param s: string to be processed
    :return: list of processed words
    """
    # detect language: english review
    if not f_lan(rw):
        return
    # lower case
    rw = f_lower(rw)
    rw = f_base(rw)
    # sentence tokenizer
    sens = sent_tokenize(rw)
    r_sens = []
    for sen in sens:
        r_sen = ' '.join(f_typo(f_punct(word_tokenize(sen))))
        if r_sen:
            r_sens.append(r_sen)
    return r_sens