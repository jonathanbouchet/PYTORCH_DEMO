import streamlit as st
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import utils.model_settings as train_model

if "model" not in st.session_state:
    st.session_state.model = None

if "num_epochs" not in st.session_state:
    st.session_state.num_epochs = None

if "train_test_split" not in st.session_state:
    st.session_state.train_test_split = None

if "batch_size" not in st.session_state:
    st.session_state.batch_size = None

if "learning_rate" not in st.session_state:
    st.session_state.learning_rate = None

if "train_loader" not in st.session_state:
    st.session_state.train_loader = None

if "test_loader" not in st.session_state:
    st.session_state.test_loader = None

st.title("Train Model")

if st.session_state["data_generated"] is None:
    st.error("choose a data template")
    with st.spinner("Redirecting to template page"):
        time.sleep(2)
        st.switch_page("templates/settings.py")
else:
    import utils.model_settings as mod_def
    import torch
    import torch.nn as nn

    num_epochs = st.sidebar.number_input(label="The number of epochs", 
                min_value=2, 
                max_value=100, 
                step=2,
                value=2,
                help="defines the number times that the learning algorithm will work through the entire training dataset.")
    st.session_state.num_epochs = num_epochs
    
    test_split = st.sidebar.number_input(label="train / test split ", 
                min_value=0.0, 
                max_value=1.0, 
                step=0.1,
                value=0.2,
                help="proportion of test samples")
    st.session_state.test_split = test_split

    batch_size = st.sidebar.number_input(label="batch size", 
                min_value=4, 
                max_value=32, 
                step=4,
                value=4,
                help="the size of the batch samples")
    st.session_state.batch_size = batch_size

    learning_rate = st.sidebar.number_input(label="learning rate", 
                min_value=0.001, 
                max_value=1.0, 
                step=0.01,
                value=0.01,
                help="""
                learning rate controls how quickly the model is adapted to the problem. 
                Smaller learning rates require more training epochs given the smaller changes made to the weights each update, 
                whereas larger learning rates result in rapid changes and require fewer training epochs.
                """)
    st.session_state.learning_rate = learning_rate
    
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        if st.session_state.features_selected:
            x0 = "Feature_0"
            x1 = "Feature_1"
            if st.session_state.data_generated is not None:
                params = st.session_state.data_parameters
                steps = int(params['n_samples'] / 100)
                x_min = 1.2*min(st.session_state.data_generated['Feature_0'])
                y_min = 1.2*min(st.session_state.data_generated['Feature_1'])
                x_max = 1.2*max(st.session_state.data_generated['Feature_0'])
                y_max = 1.2*max(st.session_state.data_generated['Feature_1'])

                df = st.session_state.data_generated
                if st.sidebar.button("Display / Run"):
                    with st.empty():
                        for i in range(steps):
                            tmp_df = st.session_state.data_generated[0:100*(i+1)]
                            fig = px.scatter(tmp_df, x="Feature_0", y="Feature_1", color='target2', range_x=[x_min, x_max], range_y=[y_min, y_max])
                            st.plotly_chart(fig, use_container_width=True, key=f"tmp_df_{i}")
                            time.sleep(0.1)
                    
                    train, test = mod_def.make_data_loader()
                    st.text("Train Dataset")
                    st.code(train.__repr__())
                    st.text("Test Dataset")
                    st.code(test.__repr__())
                    # c1, c2 = st.columns(2)
                    # c1.metric(label="train data", value=len(train), border=True)
                    # c2.metric(label="test data", value=len(test), border=True)
                    with st.spinner("running train / test ..."):
                        train_model.run(train=train, test=test)
                    if len(st.session_state.train_loss_val) > 0:
                        print(f"Epoch: {st.session_state.epoch}")
                        print(f"train loss: {st.session_state.train_loss_val}")
                        print(f"test loss: {st.session_state.test_loss_val}")
                        print(f"train acc: {st.session_state.train_acc}")
                        print(f"test acc: {st.session_state.test_acc}")
                        res_df = pd.DataFrame(data = {
                            "epoch": st.session_state.epoch, 
                            "Train Loss": st.session_state.train_loss_val, 
                            "Train Accuracy": st.session_state.train_acc,
                            "Test Loss": st.session_state.test_loss_val, 
                            "Test Accuracy": st.session_state.test_acc})
                        print(f"res df: {res_df}")
                        col3, col4 = st.columns(2)
                        with col3:
                            st.line_chart(res_df, x="epoch", y=["Train Loss","Test Loss"])
                        with col4:
                            st.line_chart(res_df, x="epoch", y=["Train Accuracy", "Test Accuracy"])
                    # train_model.run_model()
                    # display_result()
    with col2:
        st.subheader("debug", divider="red") 
        st.json(st.session_state)
