"""
IFT799 - Science des données
TP4
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

import numpy as np
from os.path import join, isfile
import pandas as pd
from scipy.stats import pearsonr
from tqdm import trange


class DataManager:
    def __init__(self, *,
                 file_path: str) -> None:
        print('Loading & processing data...')

        self.set_list = ['u1', 'u2', 'u3', 'u4', 'u5']

        cols_dict = {
            'data': ['user id', 'item id', 'rating', 'timestamp'],
            'info': ['info'],
            # 'item': ['movie id', 'movie title', 'release date', 'video release date',
            #          'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation',
            #          "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
            #          'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
            #          'Thriller', 'War', 'Western'],
            # 'genre': ['genre', 'id'],
            # 'user': ['user id', 'age', 'gender', 'occupation', 'zip code'],
            # 'occupation': ['occupation']
        }

        files_dict = {
            # 'u.data': cols_dict['data'],
            'u.info': cols_dict['info'],
            # 'u.item': cols_dict['item'],
            # 'u.genre': cols_dict['genre'],
            # 'u.user': cols_dict['user'],
            # 'u.occupation': cols_dict['occupation'],
            'u1.base': cols_dict['data'],
            'u1.test': cols_dict['data'],
            'u2.base': cols_dict['data'],
            'u2.test': cols_dict['data'],
            'u3.base': cols_dict['data'],
            'u3.test': cols_dict['data'],
            'u4.base': cols_dict['data'],
            'u4.test': cols_dict['data'],
            'u5.base': cols_dict['data'],
            'u5.test': cols_dict['data'],
            # 'ua.base': cols_dict['data'],
            # 'ua.test': cols_dict['data'],
            # 'ub.base': cols_dict['data'],
            # 'ub.test': cols_dict['data']
        }

        try:
            self.data_dict = {file: {'data': pd.read_csv(join(file_path, file),
                                                         sep='\t' if 'data' in file or
                                                                     'base' in file or
                                                                     'test' in file else '|',
                                                         names=cols,
                                                         encoding='iso8859-1')} for file, cols in files_dict.items()}
        except FileNotFoundError:
            raise FileNotFoundError(f"Please download the dataset and save it as '{file_path}' to use this script")

        self.users = self.get_info(0)
        self.items = self.get_info(1)
        self.ratings = self.get_info(2)

        self.process_data()

        print('Done!')

    def __getitem__(self, item) -> pd.DataFrame:
        return self.data_dict[item]

    def get_info(self, index: int) -> int:
        return int(self.data_dict['u.info']['data'].iloc[index, 0].split(' ')[0])

    def process_data(self) -> None:
        for set_name in self.set_list:
            df = self.data_dict[f'{set_name}.base']['data']
            user_dict = {}

            for i in range(1, self.users + 1):
                user_dict[i] = df.loc[df['user id'] == i][df.columns[1:-1]]

            self.data_dict[f'{set_name}.base']['user_dict'] = user_dict

            file_name = f'results/{set_name}.base.pearson.csv'

            if isfile(file_name):
                pearson_df = pd.read_csv(file_name)
            else:
                pearson_df = pd.DataFrame(columns=['user id 1', 'user id 2', 'pearson', 'n common'])

                for i in trange(1, self.users + 1, desc=f'Calculating pearson coefficients for {set_name}...'):
                    for j in range(1, self.users + 1):
                        if j > i:
                            pearson, n_common = self.get_similarity(set_name, i, j)
                            pearson_df = pearson_df.append({'user id 1': i,
                                                            'user id 2': j,
                                                            'pearson': pearson,
                                                            'n common': n_common},
                                                           ignore_index=True)

                pearson_df.to_csv(file_name, index=False)

            self.data_dict[f'{set_name}.base']['full_pearson'] = pearson_df

    def set_neighbors(self, n_neighbors: int) -> None:
        for set_name in self.set_list:
            pearson_df = self.data_dict[f'{set_name}.base']['full_pearson']

            self.data_dict[f'{set_name}.base']['neighbors'] = {i:
                pearson_df.loc[(pearson_df['user id 1'] == i) | (pearson_df['user id 2'] == i)]
                .sort_values(['pearson', 'n common'], ascending=[False, False])
                .head(n_neighbors)
                for i in range(1, self.users + 1)}

    def get_similarity(self, set_name: str, user_id_1: int, user_id_2: int) -> (float, int):
        data_1 = self.data_dict[f'{set_name}.base']['user_dict'][user_id_1]
        data_2 = self.data_dict[f'{set_name}.base']['user_dict'][user_id_2]

        mask_1 = self.get_mask(data_1, data_2)
        mask_2 = self.get_mask(data_2, data_1)

        masked_data_1 = self.mask_data(data_1, mask_1)
        masked_data_2 = self.mask_data(data_2, mask_2)

        if len(masked_data_1) >= 2 and masked_data_1.std() > 0 and masked_data_2.std() > 0:
            return pearsonr(masked_data_1, masked_data_2)[0], len(masked_data_1)
        else:
            return 0, len(masked_data_1)

    @staticmethod
    def get_mask(data_1: pd.DataFrame, data_2: pd.DataFrame) -> pd.DataFrame:
        return data_1['item id'].isin(data_2['item id'])

    @staticmethod
    def mask_data(data: pd.DataFrame, mask: pd.DataFrame) -> np.array:
        return data[mask].rating.to_numpy()
