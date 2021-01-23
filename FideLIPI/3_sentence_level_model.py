"""
This script performs the model training on the abstract dataset using the pretrained robera model. We use simple transformer library to train the model. We first break the abstract into sentences and assign the same label to sentence as original abstract. We then, train a model on this sentence level data. Since smaller sentences may not have enough predictive power. We train 4 models by selecting sentence above certain word count to test this hypothesis. We find that, models trained on sentence length above 10 perform the best on the validation data. Putting a sentence length filter of 6 on validation data gives us the best validation performance.

Author: Ankush Chopra (ankush01729@gmail.com)
"""

import os
import re
import torch
import spacy
import pandas as pd
import numpy as np
from operator import itemgetter
from sklearn.metrics import f1_score, confusion_matrix
from simpletransformers.classification import ClassificationModel, ClassificationArgs

# setting up the right device type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nlp = spacy.load("en_core_web_sm")


def sentence_level_data_prep(df):
    """
    This function splits the abstracts into sentences. It uses spacy for sentence tokenization.
    """

    inds = []
    sentences_extracted = []
    for abstract, ind in zip(df["text"].values, df.index):
        for i in nlp(str(abstract).replace("\n", "")).sents:
            sentences_extracted.append(str(i))
            inds.append(ind)
    sent_df = pd.DataFrame(
        {"ind": inds, "sentences_from_abstract": sentences_extracted}
    )
    return sent_df


df = pd.read_csv(r"./train.csv", header=None, names=["text", "labels"])
sentences_train = sentence_level_data_prep(df)
df.reset_index(inplace=True)
df.columns = ["ind", "text", "labels"]
sentences_train.merge(original_train[["ind", "labels"]], on="ind", how="inner")

sentences_train["sentence_length"] = sentences_train.sentences_from_abstract.map(
    lambda x: len(x.split())
)
sentences_train["label_text"] = pd.Categorical(sentences_train.labels)
sentences_train["labels"] = sentences_train.label_text.cat.codes


model_args = ClassificationArgs(
    num_train_epochs=10,
    sliding_window=True,
    fp16=False,
    use_early_stopping=True,
    reprocess_input_data=True,
    overwrite_output_dir=True,
)

# Create a ClassificationModel
model = ClassificationModel("roberta", "roberta-base", num_labels=7, args=model_args)

# We train 4 models by selecting sentences above sent_len. We save these model for 10 epochs. At the end, we select best model from these 40 saved epoch models by selecting the one doing the best on the validation set.
#
for sent_len in [0, 6, 10, 15]:
    print(sent_len)
    sentences_train_filtred = sentences_train[
        (sentences_train["sentence_length"] > sent_len)
    ]
    sentences_train_filtred.reset_index(inplace=True, drop=True)
    train = sentences_train_filtred[["sentences_from_abstract", "labels"]]

    # Optional model configuration
    output_dir = "./roberta_model_sentence_" + str(sent_len)
    best_model_dir = output_dir + "/best_model/"
    cache_dir = output_dir + "/cache/"
    print(output_dir)
    model_args = ClassificationArgs(
        cache_dir=cache_dir,
        output_dir=output_dir,
        best_model_dir=best_model_dir,
        num_train_epochs=10,
        sliding_window=True,
        fp16=False,
        use_early_stopping=True,
        reprocess_input_data=True,
        overwrite_output_dir=True,
    )

    # Create a ClassificationModel
    model = ClassificationModel(
        "roberta", "roberta-base", num_labels=7, args=model_args
    )
    # You can set class weights by using the optional weight argument
    # Train the model
    model.train_model(train)
