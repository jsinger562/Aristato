from AristatoModel import TwoLayerAristato, train, test, correct_predict_num
from feed_bert import get_loader
import numpy as np
import random
import torch
from torch import nn, optim
import matplotlib.pyplot as plt


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

def visualize_accuracy(accuracies):
    """
    Uses Matplotlib to visualize accuracy per batch.
    :return: None
    """
    x = np.arange(1, len(accuracies) + 1)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy per Epoch')
    plt.plot(x, accuracies)
    plt.show()

def test_Aristato(test_size=0.2):
    batch_size=64
    num_epoch = 25
    learning_rate = 0.01
    dataloader_train, dataloader_test = get_loader(batch_size, test_size)
    model = TwoLayerAristato()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_func = nn.MSELoss()
    losses, accuracies = train(model, dataloader_train, loss_func, optimizer, num_epoch, correct_num_func=correct_predict_num)

    loss_train, accuracy_train = test(model, dataloader_train, loss_func, correct_num_func=correct_predict_num)
    loss_test, accuracy_test = test(model, dataloader_test, loss_func, correct_num_func=correct_predict_num)
    print('Average Training Loss: {:.4f} | Average Training Accuracy: {:.4f}%'.format(loss_train, accuracy_train * 100))
    print('Average Testing Loss: {:.4f} | Average Testing Accuracy: {:.4f}%'.format(loss_test, accuracy_test * 100))

    visualize_loss(losses)
    visualize_accuracy(accuracies)


    return loss_test

def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    test_Aristato()

if __name__ == "__main__":
    main()