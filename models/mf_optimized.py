# ref: Stanislav Kuznetsov
import pandas as pd

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix


class MatrixFactorization:
    def __init__(self, name, train_data, item_ratings):
        self.name = name
        self.train_data = train_data
        self.users_amount = len(train_data)
        self.train_data_sparse = csr_matrix(self.train_data.values)
        self.item_ratings = item_ratings
        self.beta = 1.0

        self.SVD = None
        self.matrix = None
        self.model_knn = None

    def fit(self, params):
        try:
            n_components = params['n_components']
            random_state = params['random_state']

            self.SVD = TruncatedSVD(n_components=n_components, random_state=random_state)
            self.matrix = self.SVD.fit_transform(self.train_data)

            self.model_knn = NearestNeighbors(metric='cosine', algorithm='auto', n_neighbors=20, n_jobs=-1)
            self.model_knn.fit(self.matrix)

        except Exception as e:
            print(self.name + ' fit ' + str(e))

    def predict(self, user_profile, n):
        user_rated_items = user_profile[user_profile != 0].index

        user_matrix = self.SVD.transform(user_profile.values.reshape(1, -1))
        distances, indices = self.model_knn.kneighbors(user_matrix)

        my_neighbors = indices[0]
        my_neighbors_distance = distances[0]

        neighbours_profiles = self.train_data.iloc[my_neighbors]
        average_ratings = np.mean(neighbours_profiles[neighbours_profiles != 0], axis=1)

        popular_items = []
        popular_items_score = []

        for i in range(len(neighbours_profiles)):
            row = neighbours_profiles.iloc[i]
            user_average_rating = average_ratings.iloc[i]
            best_items = row[row > user_average_rating]
            popular_items.append(best_items.index)
            score = (best_items.values - user_average_rating) * (1 - my_neighbors_distance[i])
            popular_items_score.append(score)

        scores = pd.Series(np.zeros(user_profile.shape[0]), index=user_profile.index)
        for i in range(len(popular_items)):
            scores.loc[popular_items[i]] += popular_items_score[i]

        final_score = scores / self.item_ratings.sort_values(by='movieId')['sum_rating'].values
        final_score = final_score.loc[~final_score.index.isin(user_rated_items)]
        final_score.dropna(inplace=True)
        final_score = final_score[final_score != 0]
        argsort = np.argsort(final_score.values)[::-1][:n]
        recommendations = final_score[argsort].index.astype(np.int)

        return recommendations
