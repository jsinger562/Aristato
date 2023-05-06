from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class PhilosophyDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]
        

    def __getitem__(self, index):
        return self.X[index], torch.Tensor([self.Y[index]])



def get_embeddings(data, model, tokenizer):
    inputs = tokenizer(data, padding=True, truncation=True, return_tensors="pt")

    outputs = model(**inputs)

    last_hidden_state = outputs.last_hidden_state
    embeddings = torch.mean(last_hidden_state, dim=1).squeeze()
    embeddings = embeddings.detach().numpy()
    return embeddings

def write_embeddings(embeddings, label):
    file = open("data/bert.txt", "a")
    for embedding in embeddings:
        file.write(str(label) + " " + ' '.join(map(str, embedding)) + "\n")
    file.close()



def get_loader(batch_size, test_size=0.2):
    pull_size = 2
    file1 = open("data/NewNE.txt", "r")
    all_sentences = file1.read().splitlines()

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    X = get_embeddings(all_sentences[:pull_size], model, tokenizer).astype(np.float32)
    Y = np.ones(len(X))
    for i in range(1, 3):
        sentences = all_sentences[i*pull_size:(i+1)*pull_size]
        embeddings = get_embeddings(sentences, model, tokenizer)
        X_batch = embeddings.astype(np.float32)
        write_embeddings(X_batch, 1)
        Y_batch = np.ones(len(X_batch))
        X = np.concatenate((X, X_batch), axis=0)
        Y = np.hstack((Y, Y_batch))
    file1.close()
    file2 = open("data/NewRepublic.txt", "r")
    all_sentences = file2.read().splitlines()
    for i in range(3):
        sentences = all_sentences[i*pull_size:(i+1)*pull_size]
        embeddings = get_embeddings(sentences, model, tokenizer)
        X_batch = embeddings.astype(np.float32)
        Y_batch = np.zeros(len(X_batch))
        X = np.concatenate((X, X_batch), axis=0)
        Y = np.hstack((Y, Y_batch))
    file2.close()
    print(len(X))
    print(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)

    dataset_train = PhilosophyDataset(X_train, Y_train)
    dataset_test = PhilosophyDataset(X_test, Y_test)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

    return dataloader_train, dataloader_test


def visualize_loss(losses):
    """
    Uses Matplotlib to visualize loss per batch.
    :return: None
    """
    x = np.arange(1, len(losses) + 1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Training Loss per Epoch')
    plt.plot(x, losses)
    plt.show()



get_loader(10)