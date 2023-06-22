import pandas
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


def get_pca_ready_data() -> pandas.DataFrame:
    init_data = pd.read_csv('parcoursup_but.csv')

    data = init_data.loc[:, init_data.columns.isin(VARIABLES_TO_EXCLUDE)]

    PARIS_COORDINATES = [48.856, 2.3522]

    pca_ready_data = data

    pca_ready_data['coordinate1'] = pca_ready_data['Coordonnées GPS de la formation'].str.split(',').str[0].astype(
        float)
    pca_ready_data['coordinate2'] = pca_ready_data['Coordonnées GPS de la formation'].str.split(',').str[1].astype(
        float)
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

    pca_ready_data['locals_rate'] = pca_ready_data['Dont effectif des admis issus de la même académie'] / (
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


