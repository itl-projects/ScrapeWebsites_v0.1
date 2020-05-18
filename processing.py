from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import pandas as pd
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS


def get_simple_pos(tag):
    """
    natural language processing
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def lemmatize_stemming(text):
    """
    lemmatizing the text
    """
    pos = pos_tag([text])
    return WordNetLemmatizer().lemmatize(text, pos=get_simple_pos(pos[0][1]))


def preprocess(text):
    """
    lemmatizing the text
    """

    rare_words = ['apr', 'two', 'first', 'thu', 'fri', 'mon', 'tue', 'wed', 'sat', 'sun', 'month', 'day', 'year',
                  'thursday', 'january', 'february', 'march', 'april', 'june', 'july', 'august', 'september', 'october',
                  'november', 'december', 'friday', 'saturday', 'sunday', 'thursday', 'monday', 'tuesday', 'wednesday',
                  'date', 'week', 'daily', 'feb', 'september', "morning", "evening", "years", "weeks", "till", "ago",
                  'will', 'werent', 'whom', 'three', 'first', 'twice']

    result = []
    stop_free = " ".join([lemmatize_stemming(token) for token in simple_preprocess(text) if
                          token not in STOPWORDS and len(
                              token) > 2 and token not in rare_words])
    stop_free = stop_free.lower()
    return stop_free

def begin():
    """
    main execution
    """
    frame = pd.read_excel('./m.xlsx')

    frame['clean_data'] = frame.scraped_text.apply(preprocess)
    frame.to_excel('./m.xlsx')
