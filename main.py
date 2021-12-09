"""
IFT799 - Science des données
TP4
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

from os.path import join
from src.RankingPredictor import RankingPredictor
from src.DataManager import DataManager


if __name__ == '__main__':
    data_manager = DataManager(file_path=join('data', 'MovieLens-100k', 'ml-100k'))

    ranking_predictor = RankingPredictor(data_manager, dict_name='u1.base_dict')
    print(ranking_predictor(data_manager['u1.test'], k=9))
