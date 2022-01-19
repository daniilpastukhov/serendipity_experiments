import json
import time
import os

import joblib
import numpy as np
import pandas as pd
from joblib import load, dump
from tqdm import tqdm
import logging

from sklearn.model_selection import ParameterGrid

from models.ease import EASE
from models.primitive_models import PrimitiveModels
from models.knn_popular import KNNpopularity
from models.mf import MatrixFactorization
from models.autorec import AutoRec
from models.chaining import Chaining

from utils.metrics import serendipity, unexpectedness, relevance
from utils.helpers import get_movies_by_ids, get_control_items, get_movies_by_profile
from utils.parser import parse_args
from utils.logging import init_logging

import warnings

np.random.seed(42)
warnings.filterwarnings("ignore", category=RuntimeWarning)

models = {
    'KNNpopularity': KNNpopularity,
    'MatrixFactorization': MatrixFactorization,
    'EASE': EASE,
}

args = parse_args()

init_logging('chaining_gs.log', path=args.save_path)

# Datasets
DATASET = '1m'
DATA_PATH = 'data/movielens/' + DATASET + '/clean/'

ratings = pd.read_csv(DATA_PATH + 'ratings.csv')
movies = pd.read_csv(os.path.join(DATA_PATH, 'movies.csv'))
item_ratings = load(DATA_PATH + 'item_sum_dif_rating.pickle')

train_data = pd.read_csv(DATA_PATH + 'train_data.csv', index_col='userId')
test_df = pd.read_csv(DATA_PATH + 'test_data.csv', index_col='userId')
test_data, control_items = get_control_items(ratings, test_df)

ease_ratings = pd.read_csv(DATA_PATH + 'ratings.csv')
ease_test_df = pd.read_csv(DATA_PATH + 'test_data.csv', index_col='userId')
ease_test_data, _ = get_control_items(ease_ratings, user_profiles=ease_test_df)
ease_ratings, ease_control_items = get_control_items(ease_ratings, user_ids=ease_test_df.index.values)

user_embeddings = {}
for user_id, user_profile in test_data.iterrows():
    user_embeddings[user_id] = get_movies_by_profile(movies, user_profile)

logging.info('Dataset: ' + DATASET)

n = 10  # number of recommendations for each user

primitive_recommendations = None

if not os.path.isfile('cache/primitive_recommendations.joblib'):
    primitive_models = PrimitiveModels(train_data, ease_ratings, item_ratings, ease_control_items)
    primitive_recommendations = primitive_models.make_recommendations(test_data, n)
    dump(primitive_recommendations, 'cache/primitive_recommendations.joblib')
else:
    primitive_recommendations = load('cache/primitive_recommendations.joblib')


for k in [5, 15, 100]:
    for n in [20, 50, 200]:
        l = 1000
        chaining = Chaining(train_data, ease_ratings, item_ratings, ease_control_items, k, n, l)
        res = chaining.evaluate(test_data, [{'k': k}, {'n_components': n}, {'l': l}],
                                control_items,
                                user_embeddings, primitive_recommendations, movies)
        logging.debug(f'Params: k={k}, n_comp={n}, lambda={l}, '
                      f'Recall: {res[0]:.4f}, '
                      f'coverage: {res[1]:.4f}, '
                      f'novelty: {np.mean(res[2]):.4f}, '
                      f'unexpectedness: {np.mean(res[3]):.4f}, '
                      f'relevance: {np.mean(res[4]):.4f}')
        joblib.dump(res, os.path.join(args.save_path, f'res_{k}_{n}.joblib'))
