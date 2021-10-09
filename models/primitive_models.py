from sklearn.model_selection import ParameterGrid

from models.simple_knn import KNN
from models.knn_popular_optimized import KNNpopularity
from models.mf_optimized import MatrixFactorization
from models.autorec import AutoRec


models = {
    'KNNpopularity': KNNpopularity,
    'KNN': KNN,
    'MatrixFactorization': MatrixFactorization,
    'AutoRec': AutoRec
}


class PrimitiveModels:
    def __init__(self, train_data, ratings, item_ratings):
        self.primitive_models = []

        self.models = {
            'KNNpopularity': KNNpopularity('KNNpopularity', train_data, item_ratings),
            'KNN': KNN('KNN', train_data, ratings),
            'MatrixFactorization': MatrixFactorization('MatrixFactorization', train_data, item_ratings),
            'AutoRec': AutoRec('AutoRec', train_data, item_ratings)
        }

        self.primitive_grid = {
            'AutoRec': {'hidden_layer_size': [8, 16, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048], 'random_state': [11, 42, 77]},
            'KNNpopularity': {'k': [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 300, 500, 700, 1000]},
            'MatrixFactorization': {'n_components': [5, 10, 20, 50, 100, 200, 500, 1000, 2000], 'random_state': [42]},
            'KNN': {'k': [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 300, 500, 700, 1000]}
        }

        self.prepare()

    def prepare(self):
        """
        Prepare models. Use before prediction.

        :return: None.
        """
        for model_name in models:
            for grid in ParameterGrid(self.primitive_grid[model_name]):
                self.primitive_models.append(self.models[model_name])
                self.primitive_models[-1].fit(grid)

        print(f'Total {len(self.primitive_models)} primitive models.')

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
        primitive_recommendations = {}

        for user_id, user_profile in test_data.iterrows():  # iterate over test users, user_profile is a tuple
            prediction = self.predict(user_profile, n)
            primitive_recommendations[user_id] = prediction

        return primitive_recommendations
