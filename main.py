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
    liste_rankers = []

    for test_num in range(1, 6):
        ranking_predictor = RankingPredictor(data_manager, dict_name=f'u{test_num}.base_dict')
        for k in range(5, 30, 5):
            print(f"Test pour le fichier : {test_num} avec valeur de k = {k}")
            print(f" results :{ranking_predictor(data_manager['u1.test'], k=k)}")
            liste_rankers.append({'k': k,
                                  'test_num': test_num,
                                  'problems': ranking_predictor.problems,
                                  'mae': ranking_predictor.mae,
                                  'rmse': ranking_predictor.rmse,
                                  'rating_true': ranking_predictor.rating_true,
                                  'rating_pred': ranking_predictor.rating_pred})

    print("The end")
