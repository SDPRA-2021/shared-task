"""
This script performs the model training on the abstract dataset using the pretrained robera model with a classifier head. It finetunes the roberta while training for the classification task. Along with the Roberta representation of the abstract, we also use LDA vectors to train the model.
We let this run for 20 epochs and saved all the models. We selected the best models epoch when performance on the validation set stopped improving.

Author: Ankush Chopra (ankush01729@gmail.com)
"""
import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer

# setting up the device type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# reading dataset
df = pd.read_csv(r"./train.csv", header=None, names=["abs_text", "label_text"])
val_df = pd.read_csv(r"./validation.csv", header=None, names=["abs_text", "label_text"])

# reading additional features which are derived from topic models using LDA.
lda_train = pd.read_csv("./feature_lda_df_train.csv")
lda_valid = pd.read_csv("./feature_lda_df_validation.csv")

# concatinating topic vectors from LDA to the abstract dataset
df = pd.concat([df, lda_train], axis=1)
val_df = pd.concat([val_df, lda_valid], axis=1)

# # Converting the codes to appropriate categories using a dictionary
my_dict = {"CL": 0, "CR": 1, "DC": 2, "DS": 3, "LO": 4, "NI": 5, "SE": 6}


def update_cat(x):
    """
    Function to replace text labels with integer classes
    """
    return my_dict[x]


df["label_text"] = df["label_text"].apply(lambda x: update_cat(x))
val_df["label_text"] = val_df["label_text"].apply(lambda x: update_cat(x))

# Defining some key variables that will be used later on in the training
MAX_LEN = 512
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 2e-05
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


class Triage(Dataset):
    """
    This is a subclass of torch packages Dataset class. It processes input to create ids, masks and targets required for model training. 
    """

    def __init__(self, dataframe, tokenizer, max_len, text_col_name, categoty_col):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.text_col_name = text_col_name
        self.categoty_col = categoty_col
        self.col_names = list(dataframe)

    def __getitem__(self, index):
        title = str(self.data[self.text_col_name][index])
        title = " ".join(title.split())
        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "targets": torch.tensor(
                self.data[self.categoty_col][index], dtype=torch.long
            ),
            "tf_idf_feature": torch.tensor(
                self.data.loc[index, self.col_names[2:]], dtype=torch.float32
            ),
        }

    def __len__(self):
        return self.len


# dataset specifics
text_col_name = "abs_text"
category_col = "label_text"

training_set = Triage(df, tokenizer, MAX_LEN, text_col_name, category_col)
validation_set = Triage(val_df, tokenizer, MAX_LEN, text_col_name, category_col)


# data loader parameters
train_params = {"batch_size": TRAIN_BATCH_SIZE, "shuffle": True, "num_workers": 0}

test_params = {"batch_size": VALID_BATCH_SIZE, "shuffle": False, "num_workers": 0}

# creating dataloader for modelling
training_loader = DataLoader(training_set, **train_params)
val_loader = DataLoader(validation_set, **test_params)


class BERTClass(torch.nn.Module):
    """
    This is the modelling class which adds a classification layer on top of Roberta model. We finetune roberta while training for the label classification.
    """

    def __init__(self, num_class):
        super(BERTClass, self).__init__()
        self.num_class = num_class
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        self.hc_features = torch.nn.Linear(50, 128)
        self.from_bert = torch.nn.Linear(768, 128)
        self.dropout = torch.nn.Dropout(0.3)
        self.pre_classifier = torch.nn.Linear(256, 128)
        self.classifier = torch.nn.Linear(128, self.num_class)
        self.history = dict()

    def forward(self, input_ids, attention_mask, other_features):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.from_bert(pooler)
        other_feature_layer = self.hc_features(other_features)
        combined_features = torch.cat((pooler, other_feature_layer), dim=1)
        combined_features = torch.nn.ReLU()(combined_features)
        combined_features = self.dropout(combined_features)
        combined_features = self.pre_classifier(combined_features)
        output = self.classifier(combined_features)

        return output


# initializing and moving the model to the appropriate device
model = BERTClass(7)
model.to(device)

# Creating the loss function and optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


def calcuate_accu(big_idx, targets):
    """
    This function compares the predicted output with ground truth to give the count of the correct predictions.
    """
    n_correct = (big_idx == targets).sum().item()
    return n_correct


def train(epoch):
    """
    Function to train the model. This function utilizes the model initialized using BERTClass. It trains the model and provides the accuracy on the training set.
    """
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for _, data in enumerate(training_loader, 0):
        ids = data["ids"].to(device, dtype=torch.long)
        mask = data["mask"].to(device, dtype=torch.long)
        targets = data["targets"].to(device, dtype=torch.long)
        tf_idf_feature = data["tf_idf_feature"].to(device, dtype=torch.float32)

        outputs = model(ids, mask, tf_idf_feature)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accu(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        if _ % 250 == 0:
            loss_step = tr_loss / nb_tr_steps
            accu_step = (n_correct * 100) / nb_tr_examples
            print(f"Training Loss per 250 steps: {loss_step}")
            print(f"Training Accuracy per 250 steps: {accu_step}")

        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()

    print(f"The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}")
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")

    return epoch_loss, epoch_accu


def valid(model, testing_loader):
    """
    This function calculates the performance numbers on the validation set.
    """
    model.eval()
    n_correct = 0
    n_wrong = 0
    total = 0
    tr_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data["ids"].to(device, dtype=torch.long)
            mask = data["mask"].to(device, dtype=torch.long)
            targets = data["targets"].to(device, dtype=torch.long)
            tf_idf_feature = data["tf_idf_feature"].to(device, dtype=torch.float32)
            outputs = model(ids, mask, tf_idf_feature).squeeze()
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accu(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {epoch_accu}")

    return epoch_loss, epoch_accu


# path to save models at the end of the epochs
PATH = "./transformer_model_roberta_with_lda/"
if not os.path.exists(PATH):
    os.makedirs(PATH)

# variable to store the model performance at the epoch level
model.history["train_acc"] = []
model.history["val_acc"] = []
model.history["train_loss"] = []
model.history["val_loss"] = []

# model training
for epoch in range(EPOCHS):
    print("Epoch number : ", epoch)
    train_loss, train_accu = train(epoch)
    val_loss, val_accu = valid(model, val_loader)
    model.history["train_acc"].append(train_accu)
    model.history["train_loss"].append(train_loss)
    model.history["val_acc"].append(val_accu)
    model.history["val_loss"].append(val_loss)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        PATH + "/epoch_" + str(epoch) + ".bin",
    )
