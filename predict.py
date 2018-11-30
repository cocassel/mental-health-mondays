import json as j
import pandas
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
import sys


################################################## Build the model ###################################################

# Read in home improvement data and label each comment as Unrelated
home_improvement_comments = pandas.read_csv("homeimprovement_40000comments.csv")
home_improvement_comments['classification'] = 'Unrelated'
print("Home Improvement Comments: " + str(len(home_improvement_comments)))

# Read in mental health comments and label each comment as Related
mental_health_comments = pandas.read_csv("mentalhealth_40000comments_reduced.csv")
mental_health_comments['classification'] = 'Related'
print("Mental Health Comments: " + str(len(mental_health_comments)))

# Merge the two datasets
df_all_comments = pandas.concat([home_improvement_comments, mental_health_comments])

# Prepare stemmer and stopwords
stemmer = SnowballStemmer('english')
stop_words = stopwords.words("english")

# Get cleaned data (stemming, stop-word removal, etc.)
df_all_comments['cleaned'] = df_all_comments['comments'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in stop_words]).lower())

# Train-test-split with cleaned data -- use 20% for testing data
X_train, X_test, y_train, y_test = train_test_split(df_all_comments['cleaned'], df_all_comments.classification, test_size=0.2)

# Create pipeline. Use ngram_range of (1, 2) so bigrams will be considered. Use chi2 because we are performing classification
# Use k=1000 for SelectKBest based on experiment of adjusting the value. C and penality were also experimented with to reach these values.
pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                     ('chi',  SelectKBest(chi2, k=1000)),
                     ('clf', LinearSVC(C=1.0, penalty='l1', max_iter=4000, dual=False))])

# Make model
model = pipeline.fit(X_train, y_train)

# Accuracy score for testing data set
print("accuracy score: " + str(model.score(X_test, y_test)))




######################################################### Use the model ##########################################

# Read in Mental Health MOnday comments
data = pandas.read_csv("MHM_all_comments.csv")

comments = data['comments'].dropna().values.tolist()
classifications = data['classification'].dropna().values.tolist()

stripped_comments = []
for i in comments:
    stripped_comments.append(i.decode(encoding='UTF-8', errors='replace').strip())

# Lists for input that is above 10 words
comments_above_10_words = []
classifications_above_10_words = []

# Lists for input that is above 10 words and all nouns
comments_nouns = []
comments_nouns_orig = []
classifications_nouns = []

# Lists for input that is above 10 words, all nouns, and stemmed
comments_nouns_stemmed = []
comments_nouns_stemmed_orig = []
classifications_nouns_stemmed = []

# Go through all comments that were manually clasified by us and populate the three sets of lists above
for comment, classification in zip(stripped_comments, classifications):

    comment_words = comment.split()
    # Weed out really short comments -- these are generally just agreement/disagreement to other comments
    if len(comment_words) > 10:
        word_tags = nltk.pos_tag(comment_words)
        comment_words_nouns = []
        comment_words_nouns_stemmed = []
        for word, pos in word_tags:
            if pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS':
                if word not in stop_words:
                    comment_words_nouns.append(word.lower())
                    comment_words_nouns_stemmed.append(stemmer.stem(word.lower()))
        if len(comment_words_nouns_stemmed) > 0:
            comments_nouns_stemmed.append(" ".join(comment_words_nouns_stemmed))
            comments_nouns_stemmed_orig.append(comment)
            classifications_nouns_stemmed.append(classification)
        if len(comment_words_nouns) > 0:
            comments_nouns.append(" ".join(comment_words_nouns))
            comments_nouns_orig.append(comment)
            classifications_nouns.append(classification)
        comments_above_10_words.append(comment)
        classifications_above_10_words.append(classification)



# Print predictions for MANUUALLY CLASSIFIED comments

# Using inputs of above 10 words
predictions = model.predict(comments_above_10_words)
predict_df = pandas.DataFrame(
    {'comments_orig': comments_above_10_words,
     'prediction': predictions,
     'classification': classifications_above_10_words
    })
predict_df.to_csv('predict_results.csv', index=False, encoding='utf-8')

# Using inputs of above 10 words and only nouns
predictions = model.predict(comments_nouns)
predict_df = pandas.DataFrame(
    {'comments_orig': comments_nouns_orig,
     'comments_nouns': comments_nouns,
     'prediction': predictions,
     'classification': classifications_nouns
    })
predict_df.to_csv('predict_results_nouns.csv', index=False, encoding='utf-8')

# Using input of above 10 words, only nouns, and stemmed
predictions = model.predict(comments_nouns_stemmed)
predict_df = pandas.DataFrame(
    {'comments_orig': comments_nouns_stemmed_orig,
     'comments_nouns_stemmed': comments_nouns_stemmed,
     'prediction': predictions,
     'classification': classifications_nouns_stemmed
    })
predict_df.to_csv('predict_results_nouns_stemmed.csv', index=False, encoding='utf-8')



# Print ALL pridictions to CSV -- this will be used for the clustering
# Note that we have used as our input the comments above 10 words (no stemming or only keeping nouns) based on our
# experimentation yielding highest accuracy for mental health monday manually classified data.
comments_above_10_words = []

for comment in stripped_comments:
    comment_words = comment.split()
    # Weed out really short comments -- these are generally just agreement/disagreement to other comments
    if len(comment_words) > 10:
        comments_above_10_words.append(comment)

predictions = model.predict(comments_above_10_words)
predict_df = pandas.DataFrame(
    {'comments': comments_above_10_words,
     'prediction': predictions,
    })
predict_df.to_csv('predict_results_all.csv', index=False, encoding='utf-8')









