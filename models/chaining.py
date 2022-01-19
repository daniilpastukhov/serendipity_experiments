import json
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from models.ease import EASE
from models.knn_popular import KNNpopularity
from models.mf import MatrixFactorization
from utils.helpers import get_movies_by_ids, minmaxscaling
from utils.metrics import relevance, unexpectedness


class Chaining:
    def __init__(self, train_data, ease_ratings, item_ratings, ease_control_items, k, n_components, l, n=10):
        self.primitive_models = []
        self.train_data = train_data
        self.ease_ratings = ease_ratings
        self.item_ratings = item_ratings
        self.ease_control_items = ease_control_items

        self.knn = KNNpopularity('knn', train_data, item_ratings)
        self.mf = MatrixFactorization('mf', train_data, item_ratings)
        self.ease = EASE(ease_ratings, l)

        self.knn.fit({'k': k})
        self.mf.fit({'n_components': n_components,
                     'random_state': 42})

    def make_recommendations(self, test_data):
        """
        Make recommendations for users from test_data.

        :param test_data: pd.DataFrame with user profiles.
        :return: dict {user_id: recommendations}.
        """
        # todo normalize scores -> select n depending on mean score
        # todo same recs from different models
        # todo distinct recs from different models
        n = [10, 10, 10]
        recommendations = defaultdict(list)
        self.ease.prepare_recommendations(self.ease_control_items)

        for user_id, user_profile in test_data.iterrows():  # iterate over test users, user_profile is a tuple
            # EASE
            ease_scores = pd.Series(self.ease.pred[user_id], index=test_data.columns)
            ease_scores = minmaxscaling(ease_scores)
            # KNN
            knn_scores = self.knn.predict(user_profile, n[1], just_scores=True)
            knn_scores = minmaxscaling(knn_scores)
            # MF
            mf_prediction = self.mf.predict(user_profile, n[2], just_scores=True)
            mf_prediction = minmaxscaling(mf_prediction)

            combined_scores = knn_scores.combine(mf_prediction, lambda x, y: x + y, fill_value=0)\
                                        .combine(ease_scores, lambda x, y: x + y, fill_value=0)

            scores_occurences = (knn_scores > 0).astype(np.float32)\
                                                .combine((mf_prediction > 0).astype(np.float32), lambda x, y: x + y, fill_value=0)\
                                                .combine((ease_scores > 0).astype(np.float32), lambda x, y: x + y, fill_value=0)
            final_scores = combined_scores / scores_occurences

            argsort = np.argsort(final_scores)[::-1][:10]
            recommendations[user_id].extend(final_scores[argsort].index.astype(np.int))

        return recommendations

    def evaluate(self,
                 data,
                 model_params,
                 control_items,
                 user_embeddings,
                 primitive_recommendations,
                 movies):
        hit = 0  # used for recall calculation
        total_recommendations = 0
        recommendations = []
        all_recommendations = []  # used for coverage calculation
        rel = []
        unexp = []
        nov = []

        recs = self.make_recommendations(data)
        for user_id, user_profile in tqdm(data.iterrows(), total=len(data)):
            prediction = np.array(recs[user_id])
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
                [v for k, v in primitive_recommendations[user_id].items() for m in model_params if k not in json.dumps(m)]
            ).ravel()

            rel.append(np.mean(relevance(recommended_items_embeddings, user_embeddings[user_id])))
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
