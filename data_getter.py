import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def get_regions() -> pd.Series:
    """
    must return regions available in the dataset. Used for filters choice
    """
    pass


def get_social_levels() -> pd.Series:
    """
    get all unique social levels for cities. Used for filter choice
    """
    pass


def perform_clustering(n_clusters: int, scaled_data: pd.DataFrame):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    labels = kmeans.fit_predict(scaled_data)
    return labels


def assign_colors(labels):
    num_clusters = len(set(labels))
    cmap = plt.cm.get_cmap('rainbow', num_clusters)
    colors = [cmap(label) for label in labels]
    hex_colors = [rgb_to_hex(color[:3]) for color in colors]
    return hex_colors


def rgb_to_hex(rgb):
    rgb = [int(c * 255) for c in rgb]
    return '#' + ''.join([format(c, '02x') for c in rgb])


def add_diploma_centers_2d(initial_dataframe, pc2d, plotly_figure):
    diploma_positions = {}

    for index, individual in enumerate(pc2d):
        diploma = list(initial_dataframe['diploma_name'])[index]
        if diploma_positions.get(diploma) is not None:
            diploma_positions[diploma]['PC1'].append(individual[0])
            diploma_positions[diploma]['PC2'].append(individual[1])
        else:
            diploma_positions[diploma] = {'PC1': [individual[0]], 'PC2': [individual[1]]}

    diploma_means = {}
    for diploma, components in diploma_positions.items():
        mean_pc_1 = sum(components['PC1']) / len(components['PC1'])
        mean_pc_2 = sum(components['PC2']) / len(components['PC2'])
        diploma_means[diploma] = {'PC1': mean_pc_1, 'PC2': mean_pc_2}

    means_df = pd.DataFrame(diploma_means).T

    for k in range(means_df.shape[0]):
        plotly_figure.add_annotation(
            x=means_df['PC1'][k],
            y=means_df['PC2'][k],
            font={'size': 10, 'color': 'white'},
            bgcolor='black',
            text=list(means_df.index)[k].split('-')[1]
        )



 # annotation = {
        #     'x': 0,
        #     'y': 0,
        #     'text': 'Yoooaosfsdf',
        #     'font': {
        #         'size': 22,
        #         'color': 'green',
        #     },
        #     'bgcolor': 'white',
        #     'opacity': 1
        # }
        #
        # fig.update_layout(scene={'annotations': [annotation]})