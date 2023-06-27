import streamlit as st
import plotly.express as px

from function import pca_maker
from data_cleaner import get_pca_ready_data

st.set_page_config(layout="centered")
scatter_column, settings_column = st.columns((4, 1))

scatter_column.title("PCA for Multi-Dimensional Analysis demo")

option = st.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone'),
    key='contact'
)

st.write('You selected:', option)


# TODO: add region filter, remove distance from paris
# TODO: add cluster number choice
# the app will show the PCA and then the user is choosing the number of clusters
# show the contribution of different variables into component 1 and 2

pca, scaled_values, pca_data, cat_cols, pca_cols, num_data = pca_maker(get_pca_ready_data(option, option))
scatter_column.plotly_chart(
    px.scatter(
        data_frame=pca_data,
        x='PCA_1',
        y='PCA_2',
        # color=categorical_variable,  will be defined by clustering
        template="simple_white",
        hover_data=['girls_rate', 'admission_rate']
    )
)