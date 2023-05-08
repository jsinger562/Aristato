import torch
from torch import nn
import numpy as np

class TwoLayerAristato(nn.Module):

    def __init__(self, input_features=768):
        super().__init__()

        self.hidden_size = 128
        self.layer1 = nn.Linear(input_features, self.hidden_size)
        self.activation_layer = nn.Sigmoid()
        self.layer2 = nn.Linear(self.hidden_size, 1)



    def forward(self, X):
        out = self.layer1(X)
        out = self.activation_layer(out)
        out = self.layer2(out)

        return out

def train(model, dataloader, loss_func, optimizer, num_epoch, correct_num_func=None, print_info=True):
    # trains model and collects data for average losses, and average accuracy
    # much of this code is the same as that used in HW11
    average_losses = np.zeros(num_epoch)
    accuracy = np.zeros(num_epoch)
    model.train()
    for epoch in range(num_epoch):
        epoch_loss_sum = 0
        epoch_correct_num = 0
        for X, Y in dataloader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = loss_func(outputs, Y)
            loss.backward()
            optimizer.step()
            epoch_loss_sum += loss.item()*X.shape[0]
            if correct_num_func is not None:
                num_correct = correct_num_func(outputs, Y)
                epoch_correct_num += num_correct
        
        average_losses[epoch] = epoch_loss_sum/len(dataloader.dataset)
        if correct_num_func is not None:
            accuracy[epoch] = epoch_correct_num/len(dataloader.dataset)


        if print_info:
            print('Epoch: {} | Loss: {:.4f} '.format(epoch, epoch_loss_sum / len(dataloader.dataset)), end="")
            if correct_num_func:
                print('Accuracy: {:.4f}%'.format(epoch_correct_num / len(dataloader.dataset) * 100), end="")
            print()
    
    if correct_num_func is None:
        return average_losses
    else:
        return average_losses, accuracy
        

def test(model, dataloader, loss_func, correct_num_func=None):
    sum_loss = 0
    correct_predictions = 0
    model.eval()
    with torch.no_grad():
        for X, Y in dataloader:
            outputs = model(X)
            loss = loss_func(outputs, Y)
            sum_loss += loss.item()*X.shape[0]
            if correct_num_func is not None:
                correct_predictions_increase= correct_num_func(outputs, Y)
                correct_predictions += correct_predictions_increase
    if correct_num_func is None:
        return sum_loss/len(dataloader.dataset)
    else:
        return sum_loss/len(dataloader.dataset), correct_predictions/len(dataloader.dataset)
    

def correct_predict_num(logit, target):
    total = 0
    for i in range(len(logit)):
        if logit[i] > 0.5 and target[i] == 1:
            total +=1
        elif logit[i] < 0.5 and target[i] == 0:
            total += 1
    return total
