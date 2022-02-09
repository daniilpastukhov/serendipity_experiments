# ref: Stanislav Kuznetsov
import json

import pandas as pd

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from tqdm import tqdm

from utils.helpers import get_movies_by_ids
from utils.metrics import unexpectedness, relevance


class MatrixFactorization:
    def __init__(self, name, train_data, item_ratings):
        self.name = name
        self.train_data = train_data
        self.users_amount = len(train_data)
        self.train_data_sparse = csr_matrix(self.train_data.values)
        self.item_ratings = item_ratings
        self.beta = 0.0

        self.SVD = None
        self.matrix = None
        self.model_knn = None

    def fit(self, params):
        try:
            n_components = params['n_components']
            random_state = params['random_state']

            self.SVD = TruncatedSVD(n_components=n_components, random_state=random_state)
            self.matrix = self.SVD.fit_transform(self.train_data.values)

            self.model_knn = NearestNeighbors(metric='cosine', algorithm='auto', n_neighbors=20, n_jobs=-1)
            self.model_knn.fit(self.matrix)

        except Exception as e:
            print(self.name + ' fit ' + str(e))

    def predict(self, user_profile, n, just_scores=False):
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

        final_score = scores / (self.item_ratings.sort_values(by='movieId')['sum_rating'].values ** self.beta)
        final_score = final_score.loc[~final_score.index.isin(user_rated_items)]
        final_score.dropna(inplace=True)
        final_score = final_score[final_score != 0]
        if just_scores:
            return final_score
        argsort = np.argsort(final_score.values)[::-1][:n]
        recommendations = final_score[argsort].index.astype(np.int)

        return recommendations

    def evaluate(self,
                 data,
                 model_params,
                 control_items,
                 user_embeddings,
                 primitive_recommendations,
                 movies,
                 n):
        hit = 0  # used for recall calculation
        total_recommendations = 0
        recommendations = []
        all_recommendations = []  # used for coverage calculation
        rel = []
        unexp = []
        nov = []

        for user_id, user_profile in tqdm(data.iterrows(), total=len(data)):
            prediction = self.predict(user_profile, n)
            if prediction is None or prediction.ndim == 0:
                continue

            if control_items[user_id] in prediction:  # if prediction contains control item increase hit counter
                hit += 1

            recommendations.append(prediction)
            all_recommendations.extend(list(prediction))
            total_recommendations += 1

            recommended_items_embeddings = get_movies_by_ids(movies, prediction)

            if len(user_profile) == 0:  # check if user profile is empty
                continue

            primitive_recs = np.array(
                [v for k, v in primitive_recommendations[user_id].items() if k not in json.dumps(model_params)]
            ).ravel()

            rel.append(relevance(recommended_items_embeddings, user_embeddings[user_id]))
            unexp.append(unexpectedness(prediction, primitive_recs))

        if total_recommendations > 0:
            recall = hit / total_recommendations
        else:
            recall = 0

        recommendations = np.array(recommendations)
        # calculate novelty of each item
        movies_nov = pd.Series(np.zeros(len(data.columns)), index=data.columns, dtype=np.float32)
        for movie_id in data.columns:
            recommended_fraction = (recommendations == int(movie_id)).sum()
            not_interacted_fraction = (data[movie_id] == 0).sum() + 1e-10
            movies_nov[movies_nov.index == movie_id] = 1 - (recommended_fraction / not_interacted_fraction)

        movies_nov = (movies_nov - movies_nov.min()) / (movies_nov.max() - movies_nov.min())  # min-max scaling

        for recommendation in recommendations:
            nov.append(movies_nov[movies_nov.index.astype(int).isin(recommendation)].mean())

        coverage = np.unique(all_recommendations).shape[0] / self.train_data.shape[1]
        return recall, coverage, nov, unexp, rel

