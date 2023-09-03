import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

def classification_training_loop(model, X_train, y_train, X_test, y_test, loss_fn=nn.CrossEntropyLoss(), optimizer=torch.optim.SGD(), activation_function=nn.sigmoid(), epochs=1000):
    from torchmetrics import Accuracy_fn as accuracy
    """
    Args : model(Model to train), loss_fn(the loss function), optimizer, dataset(X_train, y_train, X_test, y_test), epochs(default=1000)
    My loop to train models without having to rewrite everything everytime
    Haha I'm so smart
    """

    for epoch in range(epochs):

        # Training Time !

        model.train()
        y_logits = model(X_train)
        y_pred_probs = activation_function(y_logits)
        y_pred_labels = torch.round(y_pred_probs)
        loss = loss_fn(y_logits, y_train)
        acc = accuracy(y_pred_labels, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Eval Time !

        model.eval()
        with torch.inference_mode():
            y_test_logits = model(X_test)
            y_test_pred_probs = activation_function(y_test_logits)
            y_test_pred_labels = torch.round(y_test_pred_probs)
            test_loss = loss_fn(y_test_logits, y_test)
            test_acc = accuracy(y_test_pred_labels, y_test)

        #I print out what's happening ten times during the loop
        if epoch%(epochs/10) == 0:
            print(f"Epoch: {epoch} | Loss:{loss} | Accuracy:{acc} | Test loss:{test_loss} | Test accuracy:{test_acc}")

    return loss, acc, test_loss, test_acc