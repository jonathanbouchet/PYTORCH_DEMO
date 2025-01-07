import streamlit as st

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
    pg = st.navigation([settings, display])
    pg.run()