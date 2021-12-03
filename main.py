"""
IFT799 - Science des données
TP4
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

from os.path import join
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error  # add ROOT

from src.DataManager import DataManager


if __name__ == '__main__':
    data_manager = DataManager(file_path=join('data', 'MovieLens-100k', 'ml-100k'))

