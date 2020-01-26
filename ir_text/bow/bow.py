# -*- coding: utf-8 -*-
from .tokenizer import Tokenizer
from nltk.stem.snowball import SnowballStemmer
import numpy as np
from os import listdir
from tqdm import tqdm


def createToken(document):
    tokenizer = Tokenizer([' ', '.', ',', ';', ':', '|', '(', ')', '{', '}', '[', ']', '?', '!', "'", '`', "\xa0", "«", "»", '"', "#", '\\n', '>', '<', '-', '$', '*', '£', '€', '%', '/', '@', '+', '=', '§'])
    return np.array([word.lower() for word in tokenizer.tokenize(document)])

def genStopList(language):
    stoplists = listdir('data/stoplists')
    stoplists = [stoplist for stoplist in stoplists if language in stoplist]

    stoplist_words = []
    for stoplist in stoplists :
        with open('data/stoplists/' + stoplist, "r") as stoplist :
            stoplist_words += stoplist.read().splitlines()
    return stoplist_words

def removeEmptyWords(article, stoplist_words, language):
    return article[[(word not in stoplist_words) for word in article]]

def stemming(article, language):
    stemmer = SnowballStemmer(language)
    return [stemmer.stem(word) for word in article]

def createBagOfWords(article):
    occurence, count = np.unique(article, return_counts=True)
    count_arg_sort = np.argsort(count)
    occurence = occurence[count_arg_sort]
    count = count[count_arg_sort]
    return (occurence, count)

def bow(document, language='english'):
    article = createToken(document)
    article = removeEmptyWords(article, genStopList(language), language)
    article = stemming(article, language)

    return createBagOfWords(article)


def vocabulary(documents, language='english'):
    vocabulary_list = []
    stoplist_words = genStopList(language)

    for article in tqdm(documents):
        article = createToken(article['text'])
        article = removeEmptyWords(article, stoplist_words, language)
        #print("After removing empty words   :", article)
        article = stemming(article, language)
        #print("After Stemming               :", article)
        vocabulary_list += article
        #print("Vocabulary                   :", vocabulary_list)

    return createBagOfWords(vocabulary_list)
'''
def bow_srtd_intrsc(query_bow, document_bow):
    query_mask = np.isin(query_bow[0], document_bow[0], assume_unique = True)
    document_mask = np.isin(document_bow[0], query_bow[0], assume_unique = True)

    document_sort = np.argsort(document_bow[0][document_mask])
    query_sort = np.argsort(query_bow[0][query_mask])

    return ((query_bow[0][query_mask][query_sort], query_bow[1][query_mask][query_sort]), (document_bow[0][document_mask][document_sort], document_bow[0][document_mask][document_sort]))
'''
