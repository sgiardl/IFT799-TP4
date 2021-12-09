"""
IFT799 - Science des données
TP4
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

from scipy.stats import pearsonr
from src.DataManager import DataManager
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
from tqdm import tqdm


class RankingPredictor:
    def __init__(self,
                 data_manager: DataManager,
                 dict_name: str = 'ua.base'):
        self.df = data_manager[dict_name]
        self.__pearson_values = {}  # Empty dict to store previously computed pearson_similarity values
        self.__nearest_neighbor = {}  # Empty dict to store previously computed nearest neighbors
        self.problems = []

    def __call__(self,
                 values_to_predict: pd.DataFrame,
                 k: int):

        rating_true = []
        rating_pred = []

        # Pour toutes les valeurs à prédire
        for _, row in tqdm(values_to_predict.iterrows(), total=len(values_to_predict)):
            user_id = row['user id']
            item_id = row['item id']

            pred_value = self.predict_rating(user_id, item_id, k)
            if not np.isnan(pred_value):
                rating_true.append(row['rating'])
                rating_pred.append(pred_value)

            # print(f"original : {row['rating']},   predicted : {pred_value}")
        self.rating_pred = rating_pred
        self.rating_true = rating_pred

        mae = mean_absolute_error(rating_true, rating_pred)
        rmse = (mean_squared_error(rating_true, rating_pred)) ** (1 / 2)

        self.mae = mae
        self.rmse = rmse

        return {'mae': mae, 'rmse': rmse}


    def get_knn(self, user1Id: int, itemId: int, k: int):
        """
        à faire : On calcule tous les + proches voisins, puis filtre selon l'item_id, puis retourne les k plus proches
        :param user1Id:
        :param itemId:
        :param k:
        :return:
        """
        if not user1Id in self.__nearest_neighbor:  # Si on ne connait pas encore ses plus proches voisins
            # On obtient un array de tous les autres users et leur similarité
            otherUsers = np.array([[UserIt_Id, self.pearson_similarity(user1Id, UserIt_Id)]
                                   for UserIt_Id in self.df if UserIt_Id != user1Id])
            # On retire les Users qui ont une similarité de nan
            otherUsers = otherUsers[~np.isnan(otherUsers).any(axis=1)]

            # On sort les users selon le coefficient de pearson
            sorting_indexes = otherUsers[:, 1].argsort()[::-1]
            self.__nearest_neighbor[user1Id] = otherUsers[sorting_indexes]

        # On cherche les k plus proches voisins qui ont notés l'item d'intérêt
        usersWithItem = [[us_id, us_ps] for us_id, us_ps in self.__nearest_neighbor[user1Id]
                         if itemId in self.df[us_id]['item id'].values]

        return usersWithItem[:k]

    def pearson_similarity(self, user1Id: int, user2Id: int):
        """
        Computes the pearson similarity between two users, then stores values.
        :param user1Id: User 1 id
        :param user2Id: User 2 id
        :return: The pearson similarity between the two users
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

    def predict_rating(self, userId: int, itemId: int, k: int = 10):
        # On obtient la liste des K plus proches voisins qui ont noté l'item
        userKnn = self.get_knn(userId, itemId, k)

        # On calcul le rating
        ratingSum = 0
        similarity_norm = 0

        for userIt_Id, similarity in userKnn:
            userIt_rating = self.df[userIt_Id].loc[self.df[userIt_Id]['item id'] == itemId]['rating'].mean()
            ratingSum += similarity * userIt_rating
            similarity_norm += abs(similarity)

        if similarity_norm > 0:
            if np.isnan(ratingSum / similarity_norm):
                print(f"Problème: un problème s'est produit dans le code pour le user {userId} et l'item {itemId}")
            return ratingSum / similarity_norm
        else:
            self.problems.append([userId, itemId])
            # print(f"Problème: impossible de prédir pour le User :{userId} et l'item : {itemId}")
            return np.nan  # Si aucun rating estimé, on retourne nan
