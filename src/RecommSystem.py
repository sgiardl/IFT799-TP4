"""
IFT799 - Science des données
TP4
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm


class RecommSystem:
    def __init__(self, *,
                 learning_data: dict,
                 testing_data: dict) -> None:
        self.learning_data = learning_data
        self.testing_data = testing_data

    def __call__(self, *,
                 run_name: str,
                 n_neighbors: int) -> (float, float, int):
        rating_true_list = []
        rating_pred_list = []
        n_invalid = 0

        for _, row in tqdm(self.testing_data['data'].iterrows(),
                           total=len(self.testing_data['data']),
                           desc=run_name):
            user_id = row['user id']
            item_id = row['item id']

            n_neighbors_valid = 0
            numerator = 0
            denominator = 0

            for __, row_nn in self.learning_data['neighbors'][user_id].iterrows():
                neighbor_id = row_nn['user id 1'] if row_nn['user id 1'] != user_id else row_nn['user id 2']
                df = self.learning_data['user_dict'][neighbor_id]
                neighbor_rating = df.loc[df['item id'] == item_id]['rating']

                if not neighbor_rating.empty and not np.isnan(row_nn['pearson']):
                    numerator += row_nn['pearson'] * neighbor_rating.item()
                    denominator += abs(row_nn['pearson'])
                    n_neighbors_valid += 1

                if n_neighbors_valid == n_neighbors:
                    break

            if denominator > 0:
                rating_true_list.append(row['rating'])
                rating_pred_list.append(numerator / denominator)
            else:
                n_invalid += 1

        mae = mean_absolute_error(rating_true_list, rating_pred_list)
        rmse = (mean_squared_error(rating_true_list, rating_pred_list)) ** (1 / 2)

        return mae, rmse, n_invalid
