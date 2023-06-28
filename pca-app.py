import streamlit as st
import plotly.express as px
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data_cleaner import get_pca_ready_data

# general page parameters
st.set_page_config(layout="wide")
scatter_column, settings_column = st.columns(2)

# project title
scatter_column.title("PCA for Multi-Dimensional Analysis demo")


with settings_column:
    st.write("Data filters:")
    option = st.selectbox(
        'Region?',
        ('Toute la france',),
        key='region'
    )
    pca = PCA()
    df = get_pca_ready_data(option, option)
    features = ['salary', 'admission_rate', 'girls_rate', 'locals_rate']
    X = df[features]

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(X)
    pca.fit_transform(scaled_values)
    exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)

    st.plotly_chart(
        px.area(
            x=range(1, exp_var_cumul.shape[0] + 1),
            y=exp_var_cumul,
            labels={"x": "# Components", "y": "Explained Variance"}
        )
    )


with scatter_column:
    components_number = st.selectbox(
        'Combien de composantes principales montrer?',
        (2, 3),
        key='nb_principle_components'
    )
    clusters_total = st.selectbox(
        'Combien de clusters on fait?',
        (2, 3, 4, 5, 6),
        key='nb_clusters'
    )
    df = get_pca_ready_data(option, option)
    features = ['salary', 'admission_rate', 'girls_rate', 'locals_rate']
    X = df[features]

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(X)

    if components_number == 2:
        pca = PCA(n_components=2)
        components = pca.fit_transform(scaled_values)

        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

        # replace color by clustering labels
        fig = px.scatter(components, x=0, y=1,)  # color=df['species'])

        for i, feature in enumerate(features):
            fig.add_annotation(
                ax=0, ay=0,
                axref="x", ayref="y",
                x=loadings[i, 0],
                y=loadings[i, 1],
                showarrow=True,
                arrowsize=2,
                arrowhead=2,
                xanchor="right",
                yanchor="top"
            )
            fig.add_annotation(
                x=loadings[i, 0],
                y=loadings[i, 1],
                ax=0, ay=0,
                xanchor="center",
                yanchor="bottom",
                text=feature,
                yshift=5,
            )

        scatter_column.plotly_chart(fig)

    if components_number == 3:
        pca = PCA(n_components=3)
        components = pca.fit_transform(scaled_values)
        total_var = pca.explained_variance_ratio_.sum() * 100
        fig = px.scatter_3d(
            components, x=0, y=1, z=2,
            title=f'Total Explained Variance: {total_var:.2f}%',
            labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
        )
        scatter_column.plotly_chart(fig)


