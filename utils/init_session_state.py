import streamlit as st


@st.cache_resource
def initialize_session_state() -> None:
    """initialize variables in the st.session_state

    :return _type_: _description_
    """
    if "plot_type" not in st.session_state:
        st.session_state.plot_type = None

    if "data_parameters" not in st.session_state:
        st.session_state.data_parameters = {}

    if "features_selected" not in st.session_state:
        st.session_state.features_selected = False

    if "data_submitted" not in st.session_state:
        st.session_state.data_submitted = False

    if "data_generated" not in st.session_state:
        st.session_state.data_generated = None

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

    if "train_loss_val" not in st.session_state:
        st.session_state.train_loss_val = []

    if "train_acc" not in st.session_state:
        st.session_state.train_acc = []

    if "test_loss_val" not in st.session_state:
        st.session_state.test_loss_val = []

    if "test_acc" not in st.session_state:
        st.session_state.test_acc = []

    if "epoch" not in st.session_state:
        st.session_state.epoch = []
