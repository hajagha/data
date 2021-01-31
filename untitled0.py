# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 10:35:57 2020

@author: amir
"""

import pandas as pd 
import numpy as np
from gensim.corpora.dictionary import Dictionary
import nltk

from tqdm import tqdm

import sklearn

from scipy import spatial

tqdm.pandas()

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer


from gensim.corpora.dictionary import Dictionary
from gensim.similarities import MatrixSimilarity


from gensim.models.tfidfmodel import TfidfModel



dataFrame = pd.read_csv("C:/Users/amir/Desktop/movieRecommendeer/filename.tsv" , sep = "\t")

data = dataFrame[['tconst' , 'original_title' , 'keywords']]

data = data[data.keywords.notnull()]






def preproccess(keyword) :
    
    tokenized = word_tokenize(keyword.lower())
    
    tokenized = [t for t in tokenized if t!=',' and t!= "[" and t!= ']' and t!= "'"]
    
    tokenized = [t.replace("'" , "") for t in tokenized]
    return tokenized
    




data['newKeywords'] = data['keywords'].progress_apply(lambda x:preproccess(x))


vectorizer = CountVectorizer()
corpus = data["newKeywords"].map(' '.join)

count_matrix = vectorizer.fit_transform(corpus)

cosine_sim = sklearn.metrics.pairwise.cosine_similarity(corpus)

X_train_counts = vectorizer.fit_transform(corpus)

x = X_train_counts.toarray()



