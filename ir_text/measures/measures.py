from math import sqrt, log10
import numpy as np

def dice_coef(query_words, document_words):
    return 2 * len(set(document_words).intersection(query_words))/(len(document_words)+len(query_words))

def cosine_similarity(query_bow, document_bow):
#   Three methods where implemented

#   The List & Map way :
#   Execution time / query : 0.7700
    '''
    query_intrsc = [(term, frequency) for term, frequency in zip(*query_bow) if term in document_bow[0]] 
    document_intrsc = [(term, frequency) for term, frequency in zip(*document_bow) if term in query_bow[0]]
    query_intrsc.sort()
    document_intrsc.sort()
    return sum(map(lambda q, d : q[1] * d[1], query_intrsc, document_intrsc)) / (sqrt(len(query_bow[0])) * sqrt(len(document_bow[0])))
    '''
    
#   The Numpy way
#   Execution time / query : 0.3180
    #query_len = query_bow[0].shape[0]
    #document_len = document_bow[0].shape[0]
    
    query_norm = np.linalg.norm(query_bow[1])
    document_norm = np.linalg.norm(document_bow[1])

    query_bow, document_bow = _intrsc_and_srt(query_bow, document_bow)

    return np.sum(document_bow[1] * query_bow[1]) / (query_norm * document_norm)
'''
#   The Full Map way
#   Execution time / query : 0.4594
    return sum(map(_map_cosine_similarity(document_bow),zip(*query_bow))) / (sqrt(len(query_bow[0])) * sqrt(len(document_bow[0])))
    
def _map_cosine_similarity(document_bow):
    return lambda query_tuple : query_tuple[1] * document_bow[1][np.argwhere(document_bow[0] == query_tuple[0])] if query_tuple[0] in document_bow[0] else 0
'''
def _intrsc_and_srt(query_bow, document_bow):
    query_mask = np.isin(query_bow[0], document_bow[0], assume_unique = True)
    document_mask = np.isin(document_bow[0], query_bow[0], assume_unique = True)

    document_sort = np.argsort(document_bow[0][document_mask])
    query_sort = np.argsort(query_bow[0][query_mask])

    return ((query_bow[0][query_mask][query_sort], query_bow[1][query_mask][query_sort]), (document_bow[0][document_mask][document_sort], document_bow[1][document_mask][document_sort])) 
