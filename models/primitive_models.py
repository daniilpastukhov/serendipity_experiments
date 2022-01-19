import json
from collections import defaultdict

import numpy as np
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from models.ease import EASE
from models.knn_popular import KNNpopularity
from models.mf import MatrixFactorization

models = {
    'EASE': EASE,
    'KNNpopularity': KNNpopularity,
    'MatrixFactorization': MatrixFactorization,
}


class PrimitiveModels:
    def __init__(self, train_data, ease_ratings, item_ratings, ease_control_items):
        self.primitive_models = []
        self.train_data = train_data
        self.ease_ratings = ease_ratings
        self.item_ratings = item_ratings
        self.ease_control_items = ease_control_items

        self.primitive_grid = {
            'KNNpopularity': {'k': [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 300, 500, 700, 1000]},
            'MatrixFactorization': {'n_components': [5, 10, 20, 50, 100, 200, 500, 1000, 2000], 'random_state': [42]},
            'EASE': {'l': [1.0, 10.0, 50.0, 100.0, 250.0, 500.0, 750.0, 1000.0, 2000.0, 5000.0, 10000.0]}
        }

        self.prepare()

    def prepare(self):
        """
        Prepare models. Use before prediction.

        :return: None.
        """
        for m in self.primitive_models:
            m.fit(self.primitive_grid[m.name])

    def predict(self, user_profile, n):
        """
        Make recommendations for a specific user.

        :param user_profile: One-hot encoded user profile.
        :param n: Number of recommendation. Default = 10.
        :return: List of shape (M, n), where M is a number of primitive models.
        """
        primitive_predictions = []
        for m in self.primitive_models:
            primitive_predictions.append(m.predict(user_profile, n))

        return primitive_predictions

    def make_recommendations(self, test_data, n=10):
        """
        Make recommendations for users from test_data.

        :param test_data: pd.DataFrame with user profiles.
        :param n: Number of recommendations.
        :return: dict {user_id: recommendations}.
        """
        primitive_recommendations = defaultdict(dict)

        for model_name in tqdm(models):
            for grid in ParameterGrid(self.primitive_grid[model_name]):
                # EASE API is different this if case is required
                if model_name == 'EASE':
                    model = models[model_name](self.ease_ratings, grid['l'])
                    predictions = model.prepare_recommendations(self.ease_control_items)

                    for user_id, _ in test_data.iterrows():  # iterate over test users, user_profile is a tuple
                        prediction = np.array(predictions[user_id])
                        primitive_recommendations[user_id][json.dumps(grid)] = prediction
                else:
                    model = models[model_name](model_name, self.train_data, self.item_ratings)
                    model.fit(grid)

                    for user_id, user_profile in test_data.iterrows():  # iterate over test users, user_profile is a tuple
                        prediction = model.predict(user_profile, n)
                        primitive_recommendations[user_id][json.dumps(grid)] = prediction

        return primitive_recommendations
