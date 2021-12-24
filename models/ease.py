import json
from collections import defaultdict

from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from utils.helpers import get_movies_by_ids
from utils.metrics import relevance, unexpectedness


class EASE:
    def __init__(self, ratings, l, n=10):
        self.ratings = ratings
        self.l = l
        self.n = n

        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()

        self.pred = None

    def _get_users_and_items(self, df):
        users = self.user_enc.fit_transform(df.loc[:, 'userId'])
        items = self.item_enc.fit_transform(df.loc[:, 'movieId'])
        return users, items

    def fit(self, df, lambda_: float = 0.5, implicit=False):
        """
        df: pandas.DataFrame with columns user_id, item_id and (rating)
        lambda_: l2-regularization term
        implicit: if True, ratings are ignored and taken as 1, else normalized ratings are used
        """
        users, items = self._get_users_and_items(df)
        values = np.ones(df.shape[0]) if implicit else df['rating'].to_numpy() / df['rating'].max()

        X = csr_matrix((values, (users, items)))
        # self.X = X

        G = X.T.dot(X).toarray()
        diagIndices = np.diag_indices(G.shape[0])
        G[diagIndices] += lambda_
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        B[diagIndices] = 0

        self.pred = X.dot(B)

    def predict(self, train, users, items, k):
        df = pd.DataFrame()
        items = self.item_enc.transform(items)
        dd = train.loc[train['movieId'].isin(users)]
        dd['ci'] = self.item_enc.transform(dd['movieId'])
        dd['cu'] = self.user_enc.transform(dd['userId'])
        g = dd.groupby('userId')
        for user, group in tqdm(g):
            watched = set(group['ci'])
            candidates = [item for item in items if item not in watched]
            u = group['cu'].iloc[0]
            pred = np.take(self.pred[u, :], candidates)
            res = np.argpartition(pred, -k)[-k:]
            r = pd.DataFrame({
                "userId": [user] * len(res),
                "movieId": np.take(candidates, res),
                "score": np.take(pred, res)
            }).sort_values('score', ascending=False)
            df = df.append(r, ignore_index=True)
        df['movieId'] = self.item_enc.inverse_transform(df['movieId'])
        return df

    def prepare_recommendations(self, control_items):
        self.fit(self.ratings, implicit=False, lambda_=self.l)

        pred = self.predict(self.ratings,
                            list(control_items.keys()),
                            np.unique(self.ratings['movieId']),
                            self.n)

        recommendations_ = defaultdict(list)

        for user_id, df in pred.groupby('userId'):
            recommendations_[user_id].extend(df['movieId'].values)

        return recommendations_

    def evaluate(self,
                 data,
                 model_params,
                 control_items,
                 movies,
                 user_embeddings,
                 primitive_recommendations):
        hit = 0  # used for recall calculation
        total_recommendations = 0
        user_recommendations = []
        all_recommendations = []  # used for coverage calculation
        rel = []
        unexp = []
        nov = []

        recommendations = self.prepare_recommendations(control_items)

        for user_id, user_profile in tqdm(data.iterrows(), total=len(data)):
            user_profile = user_embeddings[user_id]
            prediction = np.array(recommendations[user_id])

            if prediction is None or prediction.ndim == 0:
                continue

            if control_items[user_id] in prediction:  # if prediction contains control item increase hit counter
                hit += 1

            user_recommendations.append(prediction)
            all_recommendations.extend(list(prediction))
            total_recommendations += 1

            recommended_items = get_movies_by_ids(movies, prediction)

            if len(user_profile) == 0:  # if user profile is empty
                continue

            primitive_recs = np.array(
                [v for k, v in primitive_recommendations[user_id].items() if k not in json.dumps(model_params)]
            ).ravel()

            rel.append(relevance(recommended_items, user_embeddings[user_id]))
            unexp.append(unexpectedness(prediction, primitive_recs))

        if total_recommendations > 0:
            recall = hit / total_recommendations
        else:
            recall = 0

        user_recommendations = np.array(user_recommendations)
        # calculate novelty of each item
        movies_nov = pd.Series(np.zeros(len(data.columns)), index=data.columns, dtype=np.float32)
        for movie_id in data.columns:
            recommended_fraction = (user_recommendations == int(movie_id)).sum()
            not_interacted_fraction = (data[movie_id] == 0).sum() + 1e-10
            movies_nov[movies_nov.index == movie_id] = 1 - (recommended_fraction / not_interacted_fraction)

        movies_nov = (movies_nov - movies_nov.min()) / (movies_nov.max() - movies_nov.min())  # min-max scaling

        for recommendation in user_recommendations:
            nov.append(movies_nov[movies_nov.index.astype(int).isin(recommendation)].mean())

        coverage = np.unique(all_recommendations).shape[0] / np.unique(np.unique(self.ratings['movieId'])).shape[0]
        return recall, coverage, nov, unexp, rel
