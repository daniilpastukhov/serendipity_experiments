import numpy as np
import pandas as pd
from utils.config_parser import parse_config

config = parse_config('config/movielens100k.ini')


def get_prop_names():
    return ['ratings_mean', 'rating_std', 'features_amount', 'features_per_movie_mean', 'features_std']


def get_dict_names():
    names = get_prop_names()
    names.extend(['dataset_name', 'model', 'best_params'])
    return names


class DatasetProps:
    def __init__(self, dataset_name: str, ratings: pd.DataFrame, items: pd.DataFrame) -> None:
        self.ratings_mean = 0
        self.rating_std = 0
        self.genres_amount = 0
        self.genres_per_movie_mean = 0
        self.genres_std = 0
        self.dataset_name = dataset_name
        self.ratings = ratings
        self.items = items
        self.calculate_props()

    def calculate_props(self) -> None:
        """
        Calculate meta features.
        :return: None.
        """
        genres = self.items.filter(regex='feature\d+')
        self.ratings_mean = np.mean(self.ratings[config['ratings']['RATINGS_COL']])
        self.rating_std = np.std(self.ratings[config['ratings']['RATINGS_COL']])
        self.genres_amount = len(genres.columns)
        self.genres_per_movie_mean = np.mean(np.sum(genres, axis=1))
        self.genres_std = np.std(np.sum(genres, axis=1))

    def get_props(self, to_list=False) -> dict:
        """
        Get a dictionary of meta features.
        :param to_list: Return props as a list if set to True.
        :return: Dictionary of dataset props.
        """
        d = {
            'ratings_mean': self.ratings_mean,
            'rating_std': self.rating_std,
            'features_amount': self.genres_amount,
            'features_per_movie_mean': self.genres_per_movie_mean,
            'features_std': self.genres_std,
        }
        return d if to_list is False else np.array(list(d.values()))

    def get_dict(self, model: str, best_params: list) -> dict:
        """
        Get a dictionary containing information about dataset and its features, model name and best hyper-params.
        :param model: Model name as string.
        :param best_params: List of best params.
        :return: Dictionary with sufficient information for meta storage.
        """
        d = self.get_props()
        d.update({
            'dataset_name': self.dataset_name,
            'model': model,
            'best_params': best_params,
        })
        return d
