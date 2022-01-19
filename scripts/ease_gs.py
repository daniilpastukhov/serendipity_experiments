import os
import time
from collections import defaultdict
import logging

import pandas as pd
import numpy as np

from joblib import load, dump
from tqdm import tqdm

from models.primitive_models import PrimitiveModels
from models.ease import EASE

from utils.helpers import get_movies_by_ids, get_user_profiles, get_control_items
from utils.metrics import serendipity
from utils.logging import init_logging
from utils.parser import parse_args
from utils.helpers import get_movies_by_profile, get_movies_by_ids

np.random.seed(42)
args = parse_args()

init_logging('ease_gs.log', path=args.save_path)


def get_recommendations():
    ease = EASE()
    ease.fit(ratings, implicit=False, lambda_=param)

    pred = ease.predict(ratings, list(control_items.keys()), np.unique(ratings['movieId']), 10)

    recommendations_ = defaultdict(list)

    for user_id, df in pred.groupby('userId'):
        recommendations_[user_id].extend(df['movieId'].values)

    return recommendations_


def evaluate():
    hit = 0  # used for recall calculation
    total_recommendations = 0
    all_recommendations = []  # used for coverage calculation
    serendipity_results = []

    recommendations = get_recommendations()
    users = control_items.keys()

    for user_id in tqdm(users, total=len(users)):
        user_profile = user_embeddings[user_id]
        prediction = np.array(recommendations[user_id])
        primitive_predictions = primitive_recommendations[user_id]

        if prediction is None or prediction.ndim == 0:
            continue

        if control_items[user_id] in prediction:  # if prediction contains control item increase hit counter
            hit += 1

        all_recommendations.extend(list(prediction))
        total_recommendations += 1

        recommended_items = get_movies_by_ids(movies, prediction)

        if len(user_profile) == 0:  # if user profile is empty
            continue

        primitive_recs = np.array(list(primitive_recommendations[user_id].values()))

        rel.append(relevance(recommended_items, user_embeddings[user_id]))
        unexp.append(unexpectedness(prediction, primitive_recs))

    if total_recommendations > 0:
        recall = hit / total_recommendations
    else:
        recall = 0

    coverage = np.unique(all_recommendations).shape[0] / np.unique(np.unique(ratings['movieId'])).shape[0]
    return recall, coverage, np.nanmean(serendipity_results)


# Datasets
DATASET = '1m'
DATA_PATH = 'data/movielens/' + DATASET + '/clean/'

ratings = pd.read_csv(DATA_PATH + 'ratings.csv')
movies = pd.read_csv(os.path.join(DATA_PATH, 'movies.csv'))
item_ratings = load(DATA_PATH + 'item_sum_dif_rating.pickle')

train_data = pd.read_csv(DATA_PATH + 'train_data.csv', index_col='userId')
test_df = pd.read_csv(DATA_PATH + 'test_data.csv', index_col='userId')
test_data, _ = get_control_items(ratings, user_profiles=test_df)
ratings, control_items = get_control_items(ratings, user_ids=test_df.index.values)

# user_embeddings = get_user_profiles(ratings)

user_embeddings = {}
for user_id, user_profile in test_data.iterrows():
    user_embeddings[user_id] = get_movies_by_profile(movies, user_profile)

n = 10  # number of recommendations for each user

primitive_recommendations = None

if not os.path.isfile('cache/primitive_recommendations.joblib'):
    primitive_models = PrimitiveModels(train_data, ratings, item_ratings)
    primitive_recommendations = primitive_models.make_recommendations(test_data, n)
    dump(primitive_recommendations, 'cache/primitive_recommendations.joblib')
else:
    primitive_recommendations = load('cache/primitive_recommendations.joblib')

params = [1.0, 10.0, 50.0, 100.0, 250.0, 500.0, 750.0, 1000.0, 2000.0, 5000.0, 10000.0]
param = None
history = []

if __name__ == '__main__':
    for p in params:
        param = p
        start_time = time.time()
        logging.debug('Started: lambda={}'.format(p))
        recall, coverage, ser = evaluate()
        logging.debug('Recall: {}, coverage: {}, serendipity: {}'.format(recall, coverage, ser))
        logging.debug('Finished: lambda={}'.format(p))
        time_elapsed = time.time() - start_time
        history.append((p, recall, coverage, ser, time_elapsed))

dump(history, os.path.join(args.save_path, 'ease.joblib'))
