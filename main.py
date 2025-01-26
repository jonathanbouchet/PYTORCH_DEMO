import streamlit as st
# from templates.settings import initialize_session_state

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

st.set_page_config(layout="wide",
                   page_title="pytorch demo",
                   menu_items={
                       'Report a bug': "https://github.com/jonathanbouchet",
                       'Get help':"https://github.com/jonathanbouchet",
                       'About': "classification with pytorch demo"
    })

settings = st.Page("templates/settings.py", title="prepare data")
display = st.Page("templates/display.py", title="display data")

if __name__ == "__main__":
    # initialize_session_state()
    pg = st.navigation([settings, display])
    pg.run()