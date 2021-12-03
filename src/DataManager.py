"""
IFT799 - Science des données
TP4
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

from os.path import join
import pandas as pd


class DataManager:
    def __init__(self, file_path: str) -> None:
        cols_dict = {
            'data': ['user id', 'item id', 'rating', 'timestamp'],
            'info': ['info'],
            'item': ['movie id', 'movie title', 'release date', 'video release date',
                     'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation',
                     "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                     'Thriller', 'War', 'Western'],
            'genre': ['genre', 'id'],
            'user': ['user id', 'age', 'gender', 'occupation', 'zip code'],
            'occupation': ['occupation']
        }

        files_dict = {
            'u.data': cols_dict['data'],
            'u.info': cols_dict['info'],
            'u.item': cols_dict['item'],
            'u.genre': cols_dict['genre'],
            'u.user': cols_dict['user'],
            'u.occupation': cols_dict['occupation'],
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
            'ua.base': cols_dict['data'],
            'ua.test': cols_dict['data'],
            'ub.base': cols_dict['data'],
            'ub.test': cols_dict['data']
        }

        try:
            self.data_dict = {file: pd.read_csv(join(file_path, file),
                                                sep='\t' if 'data' in file or
                                                            'base' in file or
                                                            'test' in file else '|',
                                                names=cols,
                                                encoding='iso8859-1') for file, cols in files_dict.items()}
        except FileNotFoundError:
            raise FileNotFoundError(f"Please download the dataset and save it as '{file_path}' to use this script")

        self.users = self.get_info(0)
        self.items = self.get_info(1)
        self.ratings = self.get_info(2)

    def __getitem__(self, item) -> pd.DataFrame:
        return self.data_dict[item]

    def get_info(self, index: int) -> int:
        return int(self.data_dict['u.info'].iloc[index, 0].split(' ')[0])
