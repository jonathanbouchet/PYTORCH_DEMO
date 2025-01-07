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
                min_value=1000, 
                max_value=10000, 
                step=1000,
                value=5000,
                help="defines the number times that the learning algorithm will work through the entire training dataset.")
    st.session_state.num_epochs = num_epochs
    
    test_split = st.sidebar.number_input(label="train / test split ", 
                min_value=0.0, 
                max_value=1.0, 
                step=0.1,
                value=0.2,
                help="proportion of test samples")
    st.session_state.test_split = test_split
    
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.features_selected:
            # x0 = st.session_state.x0
            # x1 = st.session_state.x1
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
                            time.sleep(0.5)
                    train_model.run_model()
                    # display_result()
    with col2:
        st.subheader("debug", divider="red") 
        st.json(st.session_state)
