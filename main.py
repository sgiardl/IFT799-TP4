"""
IFT799 - Science des données
TP4
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

from os.path import join
from src.RankingPredictor import RankingPredictor
from sklearn.metrics import mean_absolute_error, mean_squared_error  # add ROOT

from src.DataManager import DataManager


if __name__ == '__main__':
    data_manager = DataManager(file_path=join('data', 'MovieLens-100k', 'ml-100k'))

    ranking_predictor = RankingPredictor(data_manager, dict_name='user_dict')
   # ranking_predictor(data_manager, user_id=1, item_id=1)
    ranking_predictor.get_knn(1, 2, 3)
