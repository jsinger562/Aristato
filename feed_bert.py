from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
from sklearn.model_selection import train_test_split


class PhilosophyDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]
        

    def __getitem__(self, index):
        return self.X[index], torch.Tensor([self.Y[index]])



def get_embeddings(data, model, tokenizer):
    # uses BERT model to get sequence embeddings of data
    inputs = tokenizer(data, padding=True, truncation=True, return_tensors="pt")

    outputs = model(**inputs)

    last_hidden_state = outputs.last_hidden_state
    embeddings = torch.mean(last_hidden_state, dim=1).squeeze()
    embeddings = embeddings.detach().numpy()
    return embeddings

def write_embeddings(embeddings, label):
    # writes embeddings to new txt file with label as first entry
    file = open("data/bert.txt", "a")
    for embedding in embeddings:
        file.write(str(label) + " " + ' '.join(map(str, embedding)) + "\n")
    file.close()

def feed_bert_and_write():
    # uses the two functions above to get and write embeddings to new file
    pull_size = 500
    file1 = open("data/NewNE.txt", "r")
    all_sentences = file1.read().splitlines()

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    for i in range(12):
        sentences = all_sentences[i*pull_size:(i+1)*pull_size]
        embeddings = get_embeddings(sentences, model, tokenizer)
        X_batch = embeddings.astype(np.float32)
        write_embeddings(X_batch, 1)
    file1.close()
    file2 = open("data/NewRepublic.txt", "r")
    all_sentences = file2.read().splitlines()
    for i in range(12):
        sentences = all_sentences[i*pull_size:(i+1)*pull_size]
        embeddings = get_embeddings(sentences, model, tokenizer)
        X_batch = embeddings.astype(np.float32)
        write_embeddings(X_batch, 0)
    file2.close()

def get_loader(batch_size, test_size=0.1):
    # uses newly written file with embeddings and labels to get data_loader for training
    # much of the code here is the same as in HW11
    data = np.loadtxt("data/bert.txt")
    X, Y = data[:, 1:], data[:, 0]
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
    print(len(Y_test))
    print(np.sum(Y_test))

    dataset_train = PhilosophyDataset(X_train, Y_train)
    dataset_test = PhilosophyDataset(X_test, Y_test)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

    return dataloader_train, dataloader_test

feed_bert_and_write()
