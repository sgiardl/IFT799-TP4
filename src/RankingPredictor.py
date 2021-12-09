"""
IFT799 - Science des données
TP4
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

from scipy.stats import pearsonr
from src.DataManager import DataManager
import pandas as pd
import numpy as np


class RankingPredictor:
    def __init__(self,
                 data_manager: DataManager,
                 dict_name: str = 'user_dict'):
        self.df = data_manager[dict_name]
        self.__pearson_values = {}  # Empty dict to store previously pearson_similarity values

    def __call__(self,
                 user_id: int,
                 item_id: int):
        # Get the user's ratings
        u_ratings = self.df[user_id]

        # Appliquer l'algo K-NN

        # laisse tomber la suite
        """# We get values for each user
        for v_id, v_df in data_manager[dict_name].items():
            if v_id != user_id:  # Si ce n'est pas le user que l'on cherche
                if item_id in data_manager['user_dict'][1]['item id']:  # Si le user v a noté l'item en question
                    # On construit
        """
        print()

    def pearson_similarity(self, user1Id, user2Id):
        """
        Computes the pearson similarity between two users, then stores values.
        :param user1Id: User 1 id
        :param user2Id: User 2 id
        :return: The pearson similarity between two users
        """
        if not f"{min(user1Id, user2Id)}-{max(user1Id, user2Id)}" in self.__pearson_values:
            # On prend les ratings des deux users
            user1Ratings = self.df[user1Id]
            user2Ratings = self.df[user2Id]

            # On trouve les items communs
            inner_ratings = pd.merge(user1Ratings,
                                     user2Ratings,
                                     left_on='item id',
                                     right_on='item id',
                                     suffixes=('_1', '_2'))

            if inner_ratings.shape[0] < 2:  # On ne peut calculer la similarité Pearson sinon
                self.__pearson_values[f"{min(user1Id, user2Id)}-{max(user1Id, user2Id)}"] = np.nan

            elif (inner_ratings['rating_1'] == inner_ratings['rating_1'][0]).all() or \
                 (inner_ratings['rating_2'] == inner_ratings['rating_2'][0]).all():  # Doit avoir >=1 val différente
                self.__pearson_values[f"{min(user1Id, user2Id)}-{max(user1Id, user2Id)}"] = np.nan

            else:
                ps, _ = pearsonr(inner_ratings['rating_1'], inner_ratings['rating_2'])
                self.__pearson_values[f"{min(user1Id, user2Id)}-{max(user1Id, user2Id)}"] = ps

        return self.__pearson_values[f"{min(user1Id, user2Id)}-{max(user1Id, user2Id)}"]
