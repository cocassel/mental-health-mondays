import pandas
import csv
import gensim
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.cluster import KMeansClusterer
from nltk.stem import PorterStemmer
from collections import Counter
from sklearn import cluster
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import re

# intialize stemmer
stemmer = PorterStemmer()


# Function takes in data and builds the word2vec model
def createModel (data):
    comments = data['comments'].dropna().values.tolist()

    tokenized_stopped_corpus = []

    # We'll need to look into a better tokenizer because right now it turns "co-op" into "co" and "op" !!!!!

    tokenizer = RegexpTokenizer(r'\w+')

    # Tokenizing, removing stopwords, removing punctuation, lowercasing, and only keeping nouns
    for comment in comments:
        comment_words = tokenizer.tokenize(comment.decode('utf-8', errors='replace'))
        word_tags = nltk.pos_tag(comment_words)
        comment_words_stops_removed = []
        for word, pos in word_tags:
            if word.lower() not in stopwords.words('english'):
                if pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS':
                    comment_words_stops_removed.append(word.lower())
                    #stemmed_word = stemmer.stem(word.lower())
                    #post_words_stops_removed.append(stemmed_word)
        tokenized_stopped_corpus.append(comment_words_stops_removed)
    word2vec_model = gensim.models.Word2Vec(tokenized_stopped_corpus, min_count=1, size=200)

    return word2vec_model


# Function takes in a word2vec model and creates clusters. Then it prints the clusters to CSV
def k_means_cluster(word2vec_model):
    vocab = list(word2vec_model.wv.vocab)
    X = word2vec_model[vocab]
    num_clusters = 50
    clusterer = KMeansClusterer(num_clusters, distance=nltk.cluster.util.euclidean_distance, repeats=25)
    assigned_clusters = clusterer.cluster(X, assign_clusters=True)

    dict = {}
    for index in range(num_clusters):
        cluster_list = list()
        for i, word in enumerate(vocab):
            if assigned_clusters[i] == index:
                cluster_list.append(word)
        dict[index] = cluster_list

    with open('clustering_result.csv', 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in dict.items():
            writer.writerow([key, value])






# Read in mental health monday data predictions from csv
data = pandas.read_csv("predict_results.csv")

# Only Male data
data_predicted_related = data.loc[data['prediction'] == 'Related']
model = createModel(data_predicted_related)
k_means_cluster(model)
