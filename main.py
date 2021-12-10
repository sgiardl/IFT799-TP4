"""
IFT799 - Science des données
TP4
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

import numpy as np
from os.path import join
import pandas as pd

from src.DataManager import DataManager
from src.RecommSystem import RecommSystem


if __name__ == '__main__':
    n_neighbors_list = list(range(5, 11, 5))
    set_list = ['u1', 'u2', 'u3', 'u4', 'u5']
    col_list = ['k', 'u1', 'u2', 'u3', 'u4', 'u5', 'mean', 'std']

    mae_df = pd.DataFrame(columns=col_list)
    rmse_df = pd.DataFrame(columns=col_list)
    n_invalid_df = pd.DataFrame(columns=col_list)

    data_manager = DataManager(file_path=join('data', 'MovieLens-100k', 'ml-100k'))

    for n_neighbors in n_neighbors_list:
        mae_dict = {}
        rmse_dict = {}
        n_invalid_dict = {}

        for set_name in set_list:
            recomm_system = RecommSystem(learning_data=data_manager[f'{set_name}.base'],
                                         testing_data=data_manager[f'{set_name}.test'])

            mae, rmse, n_invalid = recomm_system(run_name=f'{n_neighbors=}, {set_name=}',
                                                 n_neighbors=n_neighbors)

            mae_dict[set_name] = mae
            rmse_dict[set_name] = rmse
            n_invalid_dict[set_name] = n_invalid

        mae_dict['mean'] = sum(mae_dict.values()) / len(mae_dict)
        mae_dict['std'] = np.std(list(mae_dict.values()))

        rmse_dict['mean'] = sum(rmse_dict.values()) / len(rmse_dict)
        rmse_dict['std'] = np.std(list(rmse_dict.values()))

        n_invalid_dict['mean'] = sum(n_invalid_dict.values()) / len(n_invalid_dict)
        n_invalid_dict['std'] = np.std(list(n_invalid_dict.values()))

        mae_dict['k'] = n_neighbors
        rmse_dict['k'] = n_neighbors
        n_invalid_dict['k'] = n_neighbors

        mae_df = mae_df.append(mae_dict, ignore_index=True)
        rmse_df = rmse_df.append(rmse_dict, ignore_index=True)
        n_invalid_df = n_invalid_df.append(n_invalid_dict, ignore_index=True)

    mae_df.to_csv('results/mae.csv', index=False)
    rmse_df.to_csv('results/rmse.csv', index=False)
    n_invalid_df.to_csv('results/n_invalid.csv', index=False)

    print("Results have been saved to the 'results/' folder!")
