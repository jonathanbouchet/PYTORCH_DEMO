import streamlit as st
from sklearn import datasets
import pandas as pd
import plotly.express as px


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


def rerun_data():
    """reset only the data parameters, not the plot type"""
    print("in rerun data")
    st.session_state.data_submitted = False
    st.session_state["features_selected"] = False


def rerun_plot_type():
    """full reset"""
    print("in rerun plt_type")
    st.session_state.plot_type = None
    st.session_state.data_submitted = False
    st.session_state["features_selected"] = False
    st.session_state.data_parameters = {}
    st.session_state.data_generated = None


def display_data():
    """display toy dataset

    :return _type_: _description_
    """
    x0 = "Feature_0"
    x1 = "Feature_1"
    df = st.session_state.data_generated
    fig = px.scatter(
        df,
        x=x0,
        y=x1,
        color="target2",
        hover_data=[x0, x1],
    )
    return fig


# @st.cache_data
def make_df() -> pd.DataFrame:
    """create toy dataset based on the user parameters

    :return pd.DataFrame: dataframe
    """
    print("in make_df")
    n_features: int = 2
    params = st.session_state.data_parameters
    if st.session_state.plot_type == "blob":
        data, targets = datasets.make_classification(
            n_samples=params["n_samples"],
            n_informative=1,
            n_redundant=0,
            n_clusters_per_class=1,
            n_features=params["n_features"],
        )
        n_features = params["n_features"]
    elif st.session_state.plot_type == "circle":
        data, targets = datasets.make_circles(
            n_samples=params["n_samples"], noise=params["noise"]
        )
        n_features = 2
    elif st.session_state.plot_type == "moon":
        data, targets = datasets.make_moons(
            n_samples=params["n_samples"], noise=params["noise"]
        )
        n_features = 2
    else:
        print("not implemented")
        data, targets = None, None

    df = pd.DataFrame(data, columns=[f"Feature_{i}" for i in range(n_features)])
    # st.session_state.features = [f"Feature_{i}" for i in range(n_features)]
    df["target"] = targets
    df["target2"] = ["class_0" if val == 0 else "class_1" for val in targets]
    st.session_state.data_generated = df
    return df


# initialize_session_state()
st.title("Data preparation")
st.sidebar.text("Toy dataset generator")
st.divider()

plot_type = st.sidebar.selectbox(
    placeholder="Choose an option",
    label="chart type",
    options=["blob", "moon", "circle"],
    on_change=rerun_plot_type,
    help="choose the type of classification",
)

if plot_type is not None:
    st.session_state.plot_type = plot_type

    if plot_type == "blob":
        n_samples = st.sidebar.number_input(
            label="The number of samples",
            min_value=500,
            max_value=10000,
            step=500,
            on_change=rerun_data,
            help="The number of samples.",
        )

        n_features = st.sidebar.number_input(
            label="The total number of features",
            min_value=2,
            max_value=20,
            on_change=rerun_data,
            help="""The total number of features. These comprise n_informative informative features, 
                n_redundant redundant features, n_repeated duplicated features and n_features-n_informative-n_redundant-n_repeated useless features drawn at random.""",
        )

        class_sep = st.sidebar.number_input(
            label="The factor multiplying the hypercube size",
            min_value=1.0,
            max_value=20.0,
            step=0.5,
            on_change=rerun_data,
            help="""Larger values spread out the clusters/classes and make the classification task easier.""",
        )
        st.session_state.data_parameters = {
            "plot_type": st.session_state.plot_type,
            "n_samples": n_samples,
            "n_features": n_features,
            "class_sep": class_sep,
        }
        if st.session_state.features_selected is False:
            if st.sidebar.button(label="generate data", help="make plot"):
                st.session_state.data_submitted = True
                print(st.session_state.data_submitted)
                print(st.session_state)
                st.session_state["features_selected"] = True

                st.sidebar.button(
                    label="reset parameters",
                    help="clear parameters plot",
                    on_click=rerun_plot_type,
                )

    if plot_type in ["circle", "moon"]:
        n_samples = st.sidebar.number_input(
            label="The number of samples",
            min_value=500,
            max_value=10000,
            step=500,
            on_change=rerun_data,
            help="The number of samples.",
        )

        noise = st.sidebar.number_input(
            label="noise added to the data",
            min_value=0.0,
            max_value=0.1,
            step=0.01,
            on_change=rerun_data,
            help="""Standard deviation of Gaussian noise added to the data.""",
        )

        st.session_state.data_parameters = {
            "plot_type": st.session_state.plot_type,
            "n_samples": n_samples,
            "noise": noise,
        }
        # if st.session_state.features_selected is False:
        if st.sidebar.button(label="generate data", help="make plot"):
            st.session_state.data_submitted = True
            print(st.session_state.data_submitted)
            print(st.session_state)
            st.session_state["features_selected"] = True

            st.sidebar.button(
                label="reset parameters",
                help="clear parameters plot",
                on_click=rerun_plot_type,
            )

    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        if st.session_state.data_submitted is True:
            df = make_df()
            fig = display_data()
            st.subheader("visualization", divider="blue")
            st.plotly_chart(
                fig, use_container_width=True, key="data_plot", on_select="rerun"
            )

            st.subheader("raw dataframe", divider="blue")
            print(f"st.session_state.data_submitted :{st.session_state.data_submitted}")
            # df = make_df()
            # print(df)
            st.dataframe(df)
            # fig = display_data()
            # st.subheader("visualization", divider="blue")
            # st.plotly_chart(fig, use_container_width=True, key="data_plot", on_select="rerun")
    with col2:
        with st.container():
            st.subheader("parameters", divider="blue")
            params = st.session_state.data_parameters
            if params["plot_type"] == "blob":
                c1, c2, c3 = st.columns(3)
                c1.metric(
                    label="number of samples", value=params["n_samples"], border=True
                )
                c2.metric(
                    label="number of features", value=params["n_features"], border=True
                )
                c3.metric(
                    label="class separator", value=params["class_sep"], border=True
                )
            else:
                c1, c2 = st.columns(2)
                c1.metric(
                    label="number of samples", value=params["n_samples"], border=True
                )
                c2.metric(label="noise", value=params["noise"], border=True)
        st.subheader("debug", divider="red")
        st.json(st.session_state)
