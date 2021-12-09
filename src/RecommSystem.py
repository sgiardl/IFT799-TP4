"""
IFT799 - Science des données
TP4
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors


class RecommSystem:
    def __init__(self, *,
                 learning_data: dict,
                 testing_data: dict,
                 n_neighbors: int = 9) -> None:
        self.learning_data = learning_data
        self.testing_data = testing_data
        self.n_neighbors = n_neighbors

        self.test()

    def test(self) -> None:
        rating_true = []
        rating_pred = []

        for _, row in self.testing_data['data'].iterrows():
            user_id = row['user id']
            item_id = row['item id']
            rating_true.append(row['rating'])

            a = self.learning_data['movie_matrix']
            x = a[np.where(a[:, item_id - 1] > 0)]

            n_neighbors = len(x) if self.n_neighbors > len(x) else self.n_neighbors

            knn = NearestNeighbors(n_neighbors=n_neighbors).fit(x).kneighbors()[1]

            numerator = 0
            denominator = 0

            for nn in knn[user_id - 1]:
                numerator += self.get_similarity(user_id, x[nn, :]) * x[nn, item_id - 1]
                denominator += abs(self.get_similarity(user_id, x[nn, :]))

            rating_pred.append(numerator / denominator)

            print(f'rating_true={row["rating"]}, '
                  f'rating_pred={numerator / denominator:.2f}')

        mae = mean_absolute_error(rating_true, rating_pred)
        rmse = (mean_squared_error(rating_true, rating_pred)) ** (1 / 2)

        print(f'{mae=:.2f}, {rmse=:.2f}')

    def get_similarity(self, user_id_1: int, data_2: np.array) -> float:
        data_1 = self.get_user_info(user_id_1)

        data_2_df = pd.DataFrame(columns=['item id', 'rating'])

        for i, val in enumerate(data_2, start=1):
            if val > 0:
                data_2_df = data_2_df.append({'item id': int(i), 'rating': int(val)}, ignore_index=True)

        mask_1 = self.get_mask(data_1, data_2_df)
        mask_2 = self.get_mask(data_2_df, data_1)

        masked_data_1 = self.mask_data(data_1, mask_1)
        masked_data_2 = self.mask_data(data_2_df, mask_2)

        if len(masked_data_1) >= 2 and masked_data_1.std() > 0 and masked_data_2.std() > 0:
            return pearsonr(masked_data_1, masked_data_2)[0]
        else:
            return 0

    def get_user_info(self, user_id: int) -> np.array:
        data = self.learning_data['data'].loc[self.learning_data['data']['user id'] == user_id]
        return data[data.columns[1:-1]]

    @staticmethod
    def get_mask(data_1: pd.DataFrame, data_2: pd.DataFrame) -> pd.DataFrame:
        return data_1['item id'].isin(data_2['item id'])

    @staticmethod
    def mask_data(data: pd.DataFrame, mask: pd.DataFrame) -> np.array:
        return data[mask].rating.to_numpy()
