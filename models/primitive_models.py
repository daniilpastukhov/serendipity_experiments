from collections import defaultdict

from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from models.simple_knn import KNN
from models.knn_popular_optimized import KNNpopularity
from models.mf_optimized import MatrixFactorization
from models.autorec import AutoRec

models = {
    'AutoRec': AutoRec,
    'KNNpopularity': KNNpopularity,
    'MatrixFactorization': MatrixFactorization,
    'KNN': KNN,
}


class PrimitiveModels:
    def __init__(self, train_data, ratings, item_ratings):
        self.primitive_models = []
        self.train_data = train_data
        self.ratings = ratings
        self.item_ratings = item_ratings

        self.primitive_grid = {
            'AutoRec': {'hidden_layer_size': [8, 16, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048],
                        'random_state': [11, 42, 77]},
            'KNNpopularity': {'k': [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 300, 500, 700, 1000]},
            'MatrixFactorization': {'n_components': [5, 10, 20, 50, 100, 200, 500, 1000, 2000], 'random_state': [42]},
            'KNN': {'k': [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 300, 500, 700, 1000]}
        }

    def make_recommendations(self, test_data, n=10):
        """
        Make recommendations for users from test_data.

        :param test_data: pd.DataFrame with user profiles.
        :param n: Number of recommendations.
        :return: dict {user_id: recommendations}.
        """
        primitive_recommendations = defaultdict(list)

        for model_name in tqdm(models):
            for grid in ParameterGrid(self.primitive_grid[model_name]):
                if model_name != 'KNN':
                    model = models[model_name](model_name, self.train_data, self.item_ratings)
                else:
                    model = models[model_name](model_name, self.train_data, self.ratings)

                model.fit(grid)

                for user_id, user_profile in test_data.iterrows():  # iterate over test users, user_profile is a tuple
                    prediction = model.predict(user_profile, n)
                    primitive_recommendations[user_id].append(prediction)

        return primitive_recommendations
