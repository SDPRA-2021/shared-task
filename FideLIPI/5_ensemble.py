"""
This script does an ensemble of the 4 child model by taking the popular vote from the child models.
Ties are broken arbitrarily.

Author : Ankush Chopra (ankush01729@gmail.com)
"""
import pandas as pd
import numpy as np
import re, sys, os

import ast

from collections import Counter
from sklearn.metrics import f1_score


def ensemble_model(lda_path, sentence_model_path, vanila_model_path, tf_idf_model_path):
    """
    This function takes the 4 child model output file location as input and return the ensemble predictions as output.
    """

    # reading the data prediction by 4 child models.
    lda = pd.read_csv(lda_path)
    sentence = pd.read_csv(sentence_model_path)
    vanila = pd.read_csv(vanila_model_path)
    with open(tf_idf_model_path) as f:
        g = f.readlines()
    tf_idf = pd.DataFrame(ast.literal_eval(g[0]), columns=["predicted_labels"])

    # combining all 4 model predictions into one dataframe
    lda.reset_index(inplace=True)
    lda = pd.merge(lda, vanila, how="left", left_index=True, right_index=True)
    lda = lda[["index_x", "abs_text_x", "label_text_x", "pred", "pred_text"]]
    lda.columns = [
        "ind",
        "abs_text",
        "label_text",
        "model_with_LDA_text",
        "whole_abs_model_text",
    ]
    ddf = pd.merge(lda, sentence, how="left", on="ind")
    ddf.columns = [
        "ind",
        "abs_text",
        "label_text",
        "model_with_LDA_text",
        "whole_abs_model_text",
        "true_label",
        "sentence_model_text",
        "true_label_text",
        "pred_label_text",
    ]
    ddf = pd.concat([ddf, tf_idf], axis=1)

    # getting the final prediction by taking the max vote and breaking the ties arbitrarily
    my_dict = {"CL": 0, "CR": 1, "DC": 2, "DS": 3, "LO": 4, "NI": 5, "SE": 6}
    ddf["predicted_labels"] = ddf.predicted_labels.map(lambda x: my_dict[x])
    ddf["combined_prediction_4"] = ddf[
        [
            "whole_abs_model_text",
            "model_with_LDA_text",
            "sentence_model_text",
            "predicted_labels",
        ]
    ].values.tolist()
    ddf["selected_from_combined_prediction_4"] = ddf["combined_prediction_4"].apply(
        lambda x: Counter(x).most_common(1)[0][0]
    )

    return ddf


# f1 score calculation of ensemble model on training data.
train_out = ensemble_model(
    "./LDA_and_transformer_on_whole_abstract_train_data.csv",
    "./sentence_model_sentence_above_len6_train_prediction_model_above_len_10.csv",
    "only_transformer_on_whole_abstract_train_data.csv",
    "logistic_regression_tfidf_v2_train_predictions.txt",
)
train_f1 = f1_score(
    train_out.label_text,
    train_out.selected_from_combined_prediction_4,
    average="weighted",
)

# f1 score calculation of ensemble model on validation data.
val_out = ensemble_model(
    "./LDA_and_transformer_on_whole_abstract_val_data.csv",
    "./sentence_model_sentence_above_len6_val_prediction_model_above_len_10.csv",
    "only_transformer_on_whole_abstract_val_data.csv",
    "logistic_regression_tfidf_v2_val_predictions.txt",
)
val_f1 = f1_score(
    val_out.label_text, val_out.selected_from_combined_prediction_4, average="weighted"
)
