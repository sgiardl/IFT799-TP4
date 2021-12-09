"""
IFT799 - Science des données
TP4
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

from os.path import join

from src.DataManager import DataManager
from src.RecommSystem import RecommSystem


if __name__ == '__main__':
    data_manager = DataManager(file_path=join('data', 'MovieLens-100k', 'ml-100k'))

    recomm_system = RecommSystem(learning_data=data_manager['u1.base'],
                                 testing_data=data_manager['u1.test'])
