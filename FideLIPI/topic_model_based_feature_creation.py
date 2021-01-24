"""
This script performs features from abstract by running LDA on these. LDA gives the vectors that represent the abstracts. We use these features as input to one of the roberta model that we have built. Details of this model can be found in script 2.
Author: Sohom Ghosh
"""
import re
import os
import pandas as pd
import numpy as np
import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string


PATH = "/data/disk3/pakdd/"

# Reading data from train, test and validation files
train = pd.read_excel(PATH + "train.xlsx", sheet_name="train", header=None)
train.columns = ["text", "label"]
validation = pd.read_excel(
    PATH + "validation.xlsx", sheet_name="validation", header=None
)
validation.columns = ["text", "label"]
test = pd.read_excel(PATH + "test.xlsx", sheet_name="test", header=None)
test.columns = ["text"]


# Topic Modeling / LDA feature extraction
stop = set(stopwords.words("english"))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = "".join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


doc_clean_train = [clean(doc).split() for doc in list(train["text"])]
doc_clean_validation = [clean(doc).split() for doc in list(validation["text"])]
doc_clean_test = [clean(doc).split() for doc in list(test["text"])]

dictionary = corpora.Dictionary(doc_clean_train)

Lda = gensim.models.ldamodel.LdaModel


def tm_lda_feature_extract(doc_clean, df):
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    lmodel = Lda(doc_term_matrix, num_topics=50, id2word=dictionary, passes=50)
    feature_matrix_lda = np.zeros(shape=(df.shape[0], 50))  # as number of topics is 50
    rw = 0
    for dd in doc_clean:
        bow_vector = dictionary.doc2bow(dd)
        lis = lmodel.get_document_topics(
            bow_vector,
            minimum_probability=None,
            minimum_phi_value=None,
            per_word_topics=False,
        )
        for (a, b) in lis:
            feature_matrix_lda[rw, a] = b
        rw = rw + 1
    feature_lda_df = pd.DataFrame(feature_matrix_lda)
    return feature_lda_df


feature_lda_df_train = tm_lda_feature_extract(doc_clean_train, train)
feature_lda_df_validation = tm_lda_feature_extract(doc_clean_validation, validation)
feature_lda_df_test = tm_lda_feature_extract(doc_clean_test, test)
