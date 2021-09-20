from models.simple_knn import KNN
from models.knn_popular_optimized import KNNpopularity
from models.mf_optimized import MatrixFactorization
from models.autorec import AutoRec


class PrimitiveModels:
    def __init__(self, train_data, ratings, item_ratings):
        self.primitive_models = [
            KNN('KNN', train_data, ratings),
            KNNpopularity('KNNpopularity', train_data, item_ratings),
            MatrixFactorization('MatrixFactorization', train_data, item_ratings),
            AutoRec('AutoRec', train_data, item_ratings)
        ]

        self.primitive_grid = {
            'AutoRec': {'hidden_layer_size': 128, 'random_state': 100},
            'KNN': {'k': 5},
            'KNNpopularity': {'k': 5},
            'MatrixFactorization': {'n_components': 100, 'random_state': 42},
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
        primitive_recommendations = {}

        for user_id, user_profile in test_data.iterrows():  # iterate over test users, user_profile is a tuple
            prediction = self.predict(user_profile, n)
            primitive_recommendations[user_id] = prediction

        return primitive_recommendations
