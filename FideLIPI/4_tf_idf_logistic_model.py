"""
This script performs the model training on the abstract dataset using the features created using the TF-IDF vectorizer. Model is trained using the logistic regression algorithm which utilizes the 22K features created using 1 to 4-gram token and their tf-idf vectorized values.
Author: Sohom Ghosh
"""

import re
import os
import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


PATH = "/data/disk3/pakdd/"

# reading the input data files
train = pd.read_excel(PATH + "train.xlsx", sheet_name="train", header=None)
train.columns = ["text", "label"]
validation = pd.read_excel(
    PATH + "validation.xlsx", sheet_name="validation", header=None
)
validation.columns = ["text", "label"]
test = pd.read_excel(PATH + "test.xlsx", sheet_name="test", header=None)
test.columns = ["text"]

# TF-IDF feature creation
tfidf_model_original_v2 = TfidfVectorizer(
    ngram_range=(1, 4), min_df=0.0005, stop_words="english"
)
tfidf_model_original_v2.fit(train["text"])

# train
tfidf_df_train_original_v2 = pd.DataFrame(
    tfidf_model_original_v2.transform(train["text"]).todense()
)
tfidf_df_train_original_v2.columns = sorted(tfidf_model_original_v2.vocabulary_)

# validation
tfidf_df_valid_original_v2 = pd.DataFrame(
    tfidf_model_original_v2.transform(validation["text"]).todense()
)
tfidf_df_valid_original_v2.columns = sorted(tfidf_model_original_v2.vocabulary_)

# test
tfidf_df_test_original_v2 = pd.DataFrame(
    tfidf_model_original_v2.transform(test["text"]).todense()
)
tfidf_df_test_original_v2.columns = sorted(tfidf_model_original_v2.vocabulary_)


# Logistic Regression on tfidf_v2 (22K features)
def model(clf, train_X, train_y, valid_X, valid_y):
    clf.fit(train_X, train_y)
    pred_tr = clf.predict(train_X)
    pred_valid = clf.predict(valid_X)
    print("\nTraining F1:{}".format(f1_score(train_y, pred_tr, average="weighted")))
    print("Training Confusion Matrix \n{}".format(confusion_matrix(train_y, pred_tr)))
    print("Classification Report: \n{}".format(classification_report(train_y, pred_tr)))
    print(
        "\nValidation F1:{}".format(f1_score(valid_y, pred_valid, average="weighted"))
    )
    print(
        "Validation Confusion Matrix \n{}".format(confusion_matrix(valid_y, pred_valid))
    )
    print(
        "Classification Report: \n{}".format(classification_report(valid_y, pred_valid))
    )


lr_cnt = 0
train_X = tfidf_df_train_original_v2
valid_X = tfidf_df_valid_original_v2
test_X = tfidf_df_test_original_v2
train_y = train["label"].replace(
    {"CL": 0, "CR": 1, "DC": 2, "DS": 3, "LO": 4, "NI": 5, "SE": 6}
)
valid_y = validation["label"].replace(
    {"CL": 0, "CR": 1, "DC": 2, "DS": 3, "LO": 4, "NI": 5, "SE": 6}
)
info = "tfidf_v2_only"


print("\n ################# LR VERSION ################# " + str(lr_cnt) + "\n")

# Initializing logistic regression and training the model
lr_clf = LogisticRegression(solver="lbfgs", n_jobs=-1)
model(lr_clf, train_X, train_y, valid_X, valid_y)
params = lr_clf.get_params()
pred_tr = lr_clf.predict(train_X)
pred_valid = lr_clf.predict(valid_X)
open("lr_report_v" + str(lr_cnt) + info + ".txt", "w").write(
    str(info)
    + "\n\n"
    + str(params)
    + "\n\n lr_v"
    + str(lr_cnt)
    + ".pickle.dat"
    + "\n\n Training Confusion Matrix \n{}".format(confusion_matrix(train_y, pred_tr))
    + "\n\n Training Classification Report: \n{}".format(
        classification_report(train_y, pred_tr)
    )
    + "\n\n Validation Confusion Matrix \n{}".format(
        confusion_matrix(valid_y, pred_valid)
    )
    + "\n\n Validation Classification Report: \n{}".format(
        classification_report(valid_y, pred_valid)
    )
)
validation_predicted_lr_best = lr_clf.predict(valid_X)
repl_di = {0: "CL", 1: "CR", 2: "DC", 3: "DS", 4: "LO", 5: "NI", 6: "SE"}
open(PATH + "logistic_regression_tfidf_v2_validation_predictions.txt", "w").write(
    str([repl_di[i] for i in validation_predicted_lr_best])
)

test_predicted_lr_best = lr_clf.predict(test_X)
pd.DataFrame({"predicted_labels": [repl_di[i] for i in test_predicted_lr_best]}).to_csv(
    PATH + "logistic_reggression_on_tfidf_v2_22K_features_predicted_on_test.csv",
    index=False,
)
