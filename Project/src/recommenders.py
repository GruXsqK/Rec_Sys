import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender, bm25_weight, tfidf_weight

import os, sys
module_path = os.path.abspath(os.path.join(os.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from src.utils import prefilter_items, postfilter


class MainRecommender:

    def __init__(self, data, item_info, weighting=True, first_model_weeks=6, second_model_weeks=3, take_n_popular=7000):

        self.top_purchases = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != 999999]

        self.overall_top_purchases = data.groupby('item_id')['quantity'].count().reset_index()
        self.overall_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.overall_top_purchases = self.overall_top_purchases[self.overall_top_purchases['item_id'] != 999999]
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()
        
        self.user_buyses = data.groupby('user_id')['item_id'].unique().reset_index()
        self.user_buyses.columns=['user_id', 'actual']

        self.user_item_matrix = self._prepare_matrix(data)
        self.id_to_itemid, self.id_to_userid, \
            self.itemid_to_id, self.userid_to_id = self._prepare_dicts(self.user_item_matrix)

        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
        
        self.item_info = item_info
        
        self.first_model_weeks = first_model_weeks
        self.second_model_weeks = second_model_weeks
        
        self.val_lvl_1_size_weeks = first_model_weeks
        self.val_lvl_2_size_weeks = second_model_weeks

        self.data_train_lvl_1 = data[data['week_no'] < data['week_no'].max() - (self.val_lvl_1_size_weeks + self.val_lvl_2_size_weeks)]
        self.data_val_lvl_1 = data[(data['week_no'] >= data['week_no'].max() - (self.val_lvl_1_size_weeks + self.val_lvl_2_size_weeks)) &
                              (data['week_no'] < data['week_no'].max() - (self.val_lvl_2_size_weeks))]
        self.data_train_lvl_1 = prefilter_items(self.data_train_lvl_1, item_features=self.item_info, take_n_popular=take_n_popular)

        self.data_train_lvl_2 = self.data_val_lvl_1.copy()
        self.data_val_lvl_2 = data[data['week_no'] >= data['week_no'].max() - self.val_lvl_2_size_weeks]
        
        self.user_buyses_lvl_1 = self.data_train_lvl_1.groupby('user_id')['item_id'].unique().reset_index()
        self.user_buyses_lvl_1.columns=['user_id', 'actual']
        
        self.users_recommendations_lvl_1 = self.get_als_recommendations_users(self.user_buyses_lvl_1["user_id"], N=25)

    @staticmethod
    def _prepare_matrix(data):
        user_item_matrix = pd.pivot_table(data,
                                          index='user_id', columns='item_id',
                                          values='quantity',
                                          aggfunc='count',
                                          fill_value=0
                                          )

        user_item_matrix = user_item_matrix.astype(float)

        return user_item_matrix

    @staticmethod
    def _prepare_dicts(user_item_matrix):

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):

        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return own_recommender

    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())

        return model

    def _update_dict(self, user_id):

        if user_id not in self.userid_to_id.keys():

            max_id = max(list(self.userid_to_id.values()))
            max_id += 1

            self.userid_to_id.update({user_id: max_id})
            self.id_to_userid.update({max_id: user_id})

    def _get_similar_item(self, item_id):
        recs = self.model.similar_items(self.itemid_to_id[item_id], N=2)
        top_rec = recs[1][0]
        return self.id_to_itemid[top_rec]

    def _extend_with_top_popular(self, recommendations, N=5):

        if len(recommendations) < N:
            recommendations.extend(self.overall_top_purchases[:N])
            recommendations = recommendations[:N]

        return recommendations

    def _get_recommendations(self, user, model, N=5):

        self._update_dict(user_id=user)
        res = [self.id_to_itemid[rec[0]] for rec in model.recommend(userid=self.userid_to_id[user],
                                        user_items=csr_matrix(self.user_item_matrix).tocsr(),
                                        N=N,
                                        filter_already_liked_items=False,
                                        filter_items=[self.itemid_to_id[999999]],
                                        recalculate_user=True)]

        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_als_recommendations(self, user, N=5):

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.model, N=N)
    
    def get_als_recommendations_users(self, users, N=25):
        
        recomendations = [self.get_als_recomendations(user, N=N) for user in users]
        
        return recomendations

    def get_own_recommendations(self, user, N=5):

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.own_recommender, N=N)

    def get_similar_items_recommendation(self, user, N=5):

        top_users_purchases = self.top_purchases[self.top_purchases['user_id'] == user].head(N)

        res = top_users_purchases['item_id'].apply(lambda x: self._get_similar_item(x)).tolist()
        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_similar_users_recommendation(self, user, N=5):

        res = []

        similar_users = self.model.similar_users(self.userid_to_id[user], N=N+1)
        similar_users = [rec[0] for rec in similar_users]
        similar_users = similar_users[1:]

        for user in similar_users:
            res.extend(self.get_own_recommendations(user, N=1))

        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
