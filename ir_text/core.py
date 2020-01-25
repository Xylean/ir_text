import numpy as np
from . import bow
from . import measures
from . import evaluation
from .constants import Measures
from tqdm import tqdm


class _InvertedIndexDataStructure():
    def __init__(self):
        self.descriptors = [(["Text","with", "ID", "0", "doesn't", "exist"], [-1])]
        self.inverted = dict()

class InvertedIndex():

    def __init__(self, dataset, language='english'):
        self.dataset = dataset
        self.language = language
        self.index = _InvertedIndexDataStructure()
        self.voc = bow.vocabulary(dataset, language)

    def construct(self, idf=False):
        for term in self.voc[0] :
            self.index.inverted[term] = []

        for article in tqdm(self.dataset) :
            bag_of_words = bow.bow(article['text'])
            self.index.descriptors.append(bag_of_words) # Since index 0 is already set, each bag of word's index will be equal to is id
            for (term, frequency) in zip(*bag_of_words):
                self.index.inverted[term].append((article['id'], frequency))
        if idf :
            corpus_len = len(self.index.descriptors) - 1 #Index 0 doesn't exist
            for term in self.index.inverted:
                passage = []
                term_len = len(self.index.inverted[term])
                while self.index.inverted[term]:
                    id_doc, frequency = self.index.inverted[term].pop()
                    idf = np.log10(corpus_len/term_len)
                    passage.append((id_doc, idf * frequency))
                self.index.inverted[term] = passage

    def search(self, query, measure = Measures.DICE):
        short_list = []
        short_list_id = set()
        results = []
        query_bow = bow.bow(query['text'])
        corpus_len = len(self.index.descriptors) - 1 #Index 0 doesn't exist

        for term, _ in zip(*query_bow):
            if term in self.index.inverted :
                for article_id, frequency in self.index.inverted[term]:
                    if article_id not in short_list_id:
                        short_list.append((article_id, frequency))
                        short_list_id.add(article_id)

        for article_id, _ in short_list:
            if measure == Measures.DICE :
                results.append((article_id, measures.dice_coef(query_bow[0], self.index.descriptors[article_id][0])))
            if measure == Measures.TF :
                results.append((article_id, measures.cosine_similarity(query_bow, self.index.descriptors[article_id])))

        return sorted(results, key = lambda item : item[1], reverse = True)


class LinearIndex():

    def __init__(self, dataset, language = 'english'):
        self.dataset = dataset
        self.language = language
        self.index = dict()

    def construct(self):
        for article in self.dataset :
            self.index[article['id']] = bow.bow(article['text'])

    def search(self, query):
        results = []
        query_bow = bow.bow(query['text'])

        for article_id in self.index:
            #print("\nArticle :", self.index[article_id][0][:3])
            results.append((article_id, measures.dice_coef(query_bow[0], self.index[article_id][0])))

        return sorted(results, key = lambda item : item[1], reverse = True)
