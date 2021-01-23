# SDPRA-2021 Shared Task

### Submission by : FideLIPI
### Team Members : Ankush Chopra, Sohom Ghosh

We've built an ensemble of 4 models. These models are:

1. A classification model using Roberta pretrained model, where we finetune the model while training for the task.
2. A classification model using Roberta pretrained model combined with features created using LDA. We also try to finetune the Roberta weights along with classification layer while training for the task.
3. A classification model using where we first break the abstract into sentences, and build the model using all the sentences of length more than 10 words. We perform sentence tokenization using Spacy. Every sentence is given the same label as it's abstract. While prediction, we take the label with highest combined output probability as the prediction. We've used simple transformer library to build this model.
4. A classification model built on tf-idf features. These features consist of uni, bi, tri and four grams. We built a logistic regression model using these features.

Above 4 scores are combined to give the final prediction. Final prediction is made by popular vote, and ties are broken arbitrarily.
