import streamlit as st
import plotly.express as px
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data_cleaner import get_pca_ready_data
from data_getter import perform_clustering, assign_colors, get_regions, add_diploma_centers_2d

# general page parameters
st.set_page_config(layout='wide')

first_container = st.container()

regions = get_regions()

st.write("Data filters:")
option = st.selectbox(
    'Region?',
    ('Toute la france',),
    key='region'
)

df = get_pca_ready_data(option, option)
features = ['salary', 'admission_rate', 'girls_rate', 'locals_rate']
X = df[features]

scaler = StandardScaler()
scaled_values = scaler.fit_transform(X)

pca = PCA()
pca.fit_transform(scaled_values)
exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)

st.plotly_chart(
    px.bar(
        x=range(1, exp_var_cumul.shape[0] + 1),
        y=exp_var_cumul,
        labels={"x": "Components included", "y": "Explained Variance"}
    )
)

wcss = []
for i in range(1, 15):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(scaled_values)
    wcss.append(kmeans.inertia_)

st.write('Utilisez le methode de cout pour choisir le nombre de clusters')
st.plotly_chart(
    px.line(
        x=range(1, 15),
        y=wcss,
        labels={'x': 'Number of clusters', 'y': 'WCSS'}
    )
)


components_number = st.selectbox(
    'Combien de composantes principales montrer?',
    (2, 3, 4),
    key='nb_principle_components'
)
clusters_total = st.selectbox(
    'Combien de clusters on fait?',
    list(range(2, 15)),
    index=2,
    key='nb_clusters'
)

cluster_labels = perform_clustering(n_clusters=clusters_total, scaled_data=scaled_values)
colors = assign_colors(cluster_labels)

if components_number == 2:
    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled_values)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    fig = px.scatter(components, x=0, y=1, color=colors, height=900, width=900)

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

    add_diploma_centers_2d(df, components, fig)

    first_container.plotly_chart(fig, use_container_width=True, width=900)

if components_number == 3:
    pca = PCA(n_components=3)
    components = pca.fit_transform(scaled_values)
    total_var = pca.explained_variance_ratio_.sum() * 100
    fig = px.scatter_3d(
        components, x=0, y=1, z=2,
        title=f'Total Explained Variance: {total_var:.2f}%',
        labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'},
        color=colors
    )
    st.plotly_chart(fig)

if components_number == 4:
    st.header('Spherical horse in vacuum')
    st.image('img.png')

