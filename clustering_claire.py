import pandas
import csv
import gensim
import nltk
import sklearn
import scipy.spatial.distance
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.cluster import KMeansClusterer
from nltk.stem import PorterStemmer
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import pylab as pl

import pandas
import pylab as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import itertools
from collections import OrderedDict

# intialize stemmer
stemmer = PorterStemmer()


def getTokenizedCorpus(data):

    posts = data['Posts'].dropna().values.tolist()

    tokenized_stopped_corpus = []

    # We'll need to look into a better tokenizer because right now it turns "co-op" into "co" and "op" !!!!!

    tokenizer = RegexpTokenizer(r'\w+')

    # Tokenizing, removing stopwords, removing punctuation, lowercasing, and only keeping nouns
    for post in posts:
        post_words = tokenizer.tokenize(post.decode('utf-8', errors='replace'))
        word_tags = nltk.pos_tag(post_words)
        post_words_stops_removed = []
        for word, pos in word_tags:
            if word.lower() not in stopwords.words('english'):
                if pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS':
                    post_words_stops_removed.append(word.lower())
                    #stemmed_word = stemmer.stem(word.lower())
                    #post_words_stops_removed.append(stemmed_word)
        tokenized_stopped_corpus.append(post_words_stops_removed)


    flat_list = list(itertools.chain(*tokenized_stopped_corpus))
    sum = 0
    word_dict = dict.fromkeys(flat_list, 0)
    for word in range(0,len(flat_list)):
        sum += 1
        for text, freq in word_dict.items():
            if flat_list[word] == text:
                word_dict[text] += 1

    ordered_dict = OrderedDict(sorted(word_dict.items(), key=lambda x: x[1]))

    #print(ordered_dict)
    #print(sum)

    return tokenized_stopped_corpus

    # with open("stopwords_english.csv",'wb') as resultFile:
    #     wr = csv.writer(resultFile, dialect='excel')
    #     wr.writerows(stopwords.words('english'))
    #
    # with open("tokenize_stopped_corpus.csv",'wb') as resultFile:
    #     wr = csv.writer(resultFile, dialect='excel')
    #     wr.writerows(tokenized_stopped_corpus)
    # print(tokenized_stopped_corpus)


# Function takes in data and builds the word2vec model
def createModel(tokenized_stopped_corpus):

    word2vec_model = gensim.models.Word2Vec(tokenized_stopped_corpus, min_count=1, size=200)
    #print(word2vec_model)

    return word2vec_model


# Function takes in a word2vec model and creates clusters. Then it prints the clusters to CSV
def k_means_cluster(word2vec_model):
    vocab = list(word2vec_model.wv.vocab)
    X = word2vec_model[vocab]
    num_clusters = 5
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


# Function takes in word2vec model and creates a scatterplot of all the words in a 2 dimensional space
def plot_scatterplot_all_words(word2vec_model):

    vocab = list(word2vec_model.wv.vocab)
    X = word2vec_model[vocab]

    #print(len(X))

    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)

    df_plot = pandas.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])

    # scatter pyplot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(df_plot['x'], df_plot['y'])

    for word, pos in df_plot.iterrows():
        ax.annotate(word, pos)

    plt.show()

# Elbow Method for optimal k
    distortions = []
    K = range(1,20)
    for k in K:
        kmeanModel = KMeans(n_clusters=k*25).fit(X)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()


# Function takes in a word2vec model and a word and creates a scatterplot of similar words to the word argument
def plot_scatterplot_closest_words(word2vec_model, word):

    vector_array = np.empty((0, 200), dtype='f')
    word_labels = [word]

    # get close words
    close_words = word2vec_model.similar_by_word(word)

    # add the vector for each of the closest words to the array
    vector_array = np.append(vector_array, np.array([word2vec_model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = word2vec_model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        vector_array = np.append(vector_array, np.array([wrd_vector]), axis=0)

    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(vector_array)


    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
    plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
    plt.show()


# Read in data from csv
data = pandas.read_csv("MentalHealthMondayData.csv")
corpus = getTokenizedCorpus(data)
model = createModel(corpus)
k_means_cluster(model)
plot_scatterplot_all_words(model)
plot_scatterplot_closest_words(model, "interview")


