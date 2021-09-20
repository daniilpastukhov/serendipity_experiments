# ref: Stanislav Kuznetsov
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


class KNNpopularity:
    def __init__(self, name, train_data, item_ratings):
        self.name = name

        self.train_data = train_data
        self.users_amount = len(train_data)
        self.train_data_sparse = csr_matrix(self.train_data.values)

        self.k = 0
        self.model_knn = None

        self.item_ratings = item_ratings
        self.beta = 0.0

    def fit(self, params):
        try:
            self.k = params['k']
            if 'beta' in params:
                self.beta = params['beta']
            self.model_knn = NearestNeighbors(metric='cosine', algorithm='auto', n_neighbors=self.k, n_jobs=-1)
            self.model_knn.fit(self.train_data_sparse)

        except Exception as e:
            print(self.name + ' fit ' + str(e))

    def predict(self, user_profile, n):
        try:
            user_csr_profile = csr_matrix(user_profile)
            user_rated_items = user_profile[user_profile != 0].index

            distances, indices = self.model_knn.kneighbors(user_csr_profile)

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

            final_score = scores / (self.item_ratings.sort_values(by='movieId')['sum_rating'].values ** self.beta)
            final_score = final_score.loc[~final_score.index.isin(user_rated_items)]
            final_score.dropna(inplace=True)
            final_score = final_score[final_score != 0]
            argsort = np.argsort(final_score.values)[::-1][:n]
            recommendations = final_score[argsort].index.astype(np.int)

            return recommendations

        except Exception as e:
            # print(self.name + ' predictItemByUser ' + str(e))
            return None
