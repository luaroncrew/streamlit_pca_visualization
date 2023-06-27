import pandas as pd


VARIABLES_TO_EXCLUDE = [
        'Coordonnées GPS de la formation',
        'Effectif total des candidats pour une formation',
        'Dont effectif des candidates admises',
        'Effectif des admis en phase principale',
        'Effectif des admis en phase complémentaire',
        'Dont effectif des admis issus de la même académie',
        'diploma_name',
        'Filière de formation.1',
        'salary'
    ]
PARIS_COORDINATES = [48.856, 2.3522]


def get_pca_ready_data(
        region_filter: pd.Series = None,
        social_level_filter: str = None
) -> pd.DataFrame:
    """
    create a dataframe out of the csv file, apply filters added by user
    :param region_filter: the request
    :param social_level_filter:
    :return:
    """
    print(region_filter, social_level_filter)
    init_data = pd.read_csv('parcoursup_but.csv')
    data = init_data.loc[:, init_data.columns.isin(VARIABLES_TO_EXCLUDE)]
    pca_ready_data = data

    # variable modifications
    pca_ready_data['coordinate1'] = pca_ready_data[
        'Coordonnées GPS de la formation'
    ].str.split(',').str[0].astype(float)

    pca_ready_data['coordinate2'] = pca_ready_data[
        'Coordonnées GPS de la formation'
    ].str.split(',').str[1].astype(float)

    pca_ready_data['distance_from_Paris'] = (
        (pca_ready_data['coordinate1'] - PARIS_COORDINATES[0]) ** 2 +
        (pca_ready_data['coordinate2'] - PARIS_COORDINATES[1]) ** 2
    ) ** 0.5

    pca_ready_data['admission_rate'] = (
        pca_ready_data['Effectif des admis en phase principale'] +
        pca_ready_data['Effectif des admis en phase complémentaire']
    ) / pca_ready_data['Effectif total des candidats pour une formation']

    pca_ready_data['girls_rate'] = pca_ready_data['Dont effectif des candidates admises'] / (
        pca_ready_data['Effectif des admis en phase principale'] +
        pca_ready_data['Effectif des admis en phase complémentaire']
    )

    pca_ready_data['locals_rate'] = pca_ready_data[
        'Dont effectif des admis issus de la même académie'
    ] / (
        pca_ready_data['Effectif des admis en phase principale'] +
        pca_ready_data['Effectif des admis en phase complémentaire']
    )

    pca_ready_data = pca_ready_data.loc[
        :,
        ~pca_ready_data.columns.isin([
            "Effectif des admis en phase principale",
            "Effectif total des candidats pour une formation",
            "Effectif des admis en phase complémentaire",
            'Coordonnées GPS de la formation',
            'diploma_name',
            'Filière de formation.1',
            'Dont effectif des candidates admises',
            'Dont effectif des admis issus de la même académie',
            'coordinate1',
            'coordinate2'
        ])
    ]

    return pca_ready_data


