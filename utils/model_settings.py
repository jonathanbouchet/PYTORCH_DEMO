import streamlit as st
import time
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import torch.nn as nn
import numpy as np

if "train_loss_val" not in st.session_state:
    st.session_state.train_loss_val = []

if "train_loss_epoch" not in st.session_state:
    st.session_state.train_loss_epoch = []

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

def make_model():
    # Make device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device
    class CircleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_1 = nn.Linear(in_features=2, out_features=100)
            self.layer_2 = nn.Linear(in_features=100, out_features=50)
            self.layer_3 = nn.Linear(in_features=50, out_features=10)
            self.layer_4 = nn.Linear(in_features=10, out_features=1)
            self.relu = nn.ReLU() # <- add in ReLU activation function
            self.tanh = nn.Tanh()
            self.sigmoid = nn.Sigmoid()
            self.leakyRelu = nn.LeakyReLU()

        def forward(self, x):
            # Intersperse the ReLU activation function between layers
            return self.layer_4(self.leakyRelu(self.layer_3(self.leakyRelu(self.layer_2(self.leakyRelu(self.layer_1(x)))))))

    model = CircleModel().to(device)
    return model

def run_model():
    # Make device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with st.spinner("creating model"):
        model = make_model()
        time.sleep(1)
    # Setup loss and optimizer 
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    # Turn data into tensors
    X = np.array([st.session_state.data_generated['Feature_0'], 
                  st.session_state.data_generated["Feature_1"]]
    )
    X = X.transpose()
    y = np.array(st.session_state.data_generated['target'])
    X_torch = torch.from_numpy(X).type(torch.float)
    y_torch = torch.from_numpy(y).type(torch.float)

    print(X_torch.shape, y_torch.shape)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_torch,
        y_torch,
        test_size=st.session_state.train_test_split,
        random_state=42) # make the random split reproducible

    len(X_train), len(X_test), len(y_train), len(y_test)

    # Fit the model
    torch.manual_seed(42)
    epochs = st.session_state.num_epochs
    train_loss_arr = []
    test_loss_arr = []
    train_acc_arr = []
    test_acc_arr = []

    # Put all data on target device
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    st.session_state.pytorch_data = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}

    for epoch in range(epochs):
        # 1. Forward pass
        y_logits = model(X_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits)) # logits -> prediction probabilities -> prediction labels
        
        # 2. Calculate loss and accuracy
        loss = loss_fn(y_logits, y_train) # BCEWithLogitsLoss calculates loss using logits
        acc = accuracy_fn(y_true=y_train, 
                        y_pred=y_pred)
        
        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        print(f"epoch :{epoch}, train loss:{loss}")

        st.session_state.train_loss_val.append(loss)
        st.session_state.train_loss_epoch.append(epoch)

        ### Testing
        model.eval()
        with torch.inference_mode():
            # 1. Forward pass
            test_logits = model(X_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits)) # logits -> prediction probabilities -> prediction labels
            # 2. Calculate loss and accuracy
            test_loss = loss_fn(test_logits, y_test)
            test_acc = accuracy_fn(y_true=y_test,
                                    y_pred=test_pred)
            
            train_loss_arr.append(loss)
            test_loss_arr.append(test_loss)
            train_acc_arr.append(acc)
            test_acc_arr.append(test_acc)

        # Print out what's happening
        if (epoch + 1) % 1000 == 0:
            st.text(f"Epoch: {epoch} | Loss: {loss:.5f} | Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f} | Test Accuracy: {test_acc:.2f}%")

        # save model in session_state
        st.session_state.model = model