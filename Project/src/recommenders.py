import pandas as pd
import numpy as np

import catboost as catb

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

    
class CastomRecommender:
 
    def __init__(self,
                 data, 
                 items_features, 
                 users_features,
                 weighting=False,
                 weeks_model_lv1=26,
                 weeks_model_lv2=6,
                 recs_limit_model_lv1=200,
                 prefilter_params = {'n_popular_limit':5000,
                                     'upper_popularity_limit':1,
                                     'lower_popularity_limit':0.005,
                                     'lower_price_limit':1,
                                     'upper_price_limit':300,
                                     'min_dep_assortment':100},
                 als_params = {'n_factors':30,
                               'regularization':0.001,
                               'iterations':35}):
        
        self.data = data
        self.users_features = users_features
        self.items_features = items_features

        self.weeks_model_lv1 = weeks_model_lv1
        self.weeks_model_lv2 = weeks_model_lv2   
        
        self.weighting = weighting
        self.prefilter_params = prefilter_params
        self.als_params = als_params
        self.recs_limit_model_lv1 = recs_limit_model_lv1
        
        self.data_items = None
        self.top_purchases = None
        self.top_popular = None
        self.user_item_matrix = None
        self.id_to_itemid = None
        self.id_to_userid = None
        self.itemid_to_id = None
        self.userid_to_id = None
        
        self._add_item_features()
        self._add_user_features()
        self._prepare_data()
        self._prepare_user_item_matrix()
        
        self._add_top_popular()
        self._add_top_purchases()
        self._add_top_purchases_by_user()
        self._add_own_recs()
        self._add_als_recs(**als_params)
        self._add_basic_recs()
        self.data_matrix=self._prepare_data_matrix(data=self.users_features,
                                                   column_recommended='basic_recommender',
                                                   column_purchases_train='purchases_train',
                                                   users_features=self.users_features)
        self._fit_ranking()
        
 
    def _prepare_user_item_matrix(self):
        
        user_item_matrix = pd.pivot_table(self.data_items,
                                          index='user_id',
                                          columns='item_id',
                                          values='quantity',
                                          aggfunc='count',
                                          fill_value=0)

        self.user_item_matrix = user_item_matrix.astype(float)
        self.user_item_matrix = csr_matrix(self.user_item_matrix).tocsr()
        
        if self.weighting:
            self.user_item_matrix = tfidata_weight(self.user_item_matrix.T).T
        
        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        self.id_to_itemid = dict(zip(matrix_itemids, itemids))
        self.id_to_userid = dict(zip(matrix_userids, userids))

        self.itemid_to_id = dict(zip(itemids, matrix_itemids))
        self.userid_to_id = dict(zip(userids, matrix_userids))
        
        self.users_features['id'] = self.users_features['user_id'].map(self.userid_to_id)
        self.items_features['id'] = self.items_features['item_id'].map(self.itemid_to_id)   


    def _prepare_data_matrix(self, 
                             data=None, 
                             column_recommended=None, 
                             column_purchases_train=None, 
                             users_features=None, 
                             items_features=None):
        
        def normalize_data_column(data, normalize_column):
            user_id_item_list_dict = data.set_index('user_id')[normalize_column].to_dict()
            data_matrix = pd.DataFrame()
            for user_id in user_id_item_list_dict.keys():
                data_matrix = pd.concat((data_matrix, pd.DataFrame({'user_id':user_id, 'item_id':user_id_item_list_dict[user_id]})), axis=0)
            return data_matrix
        
        data_matrix = normalize_data_column(data=data, normalize_column=column_recommended)
        
        user_skip_cols = ['top_popular', 
                          'top_purchases', 
                          'top_purchases_by_user', 
                          'own_recommender', 
                          'als_recommender', 
                          'basic_recommender' , 
                          'purchases_train', 
                          'id']
        
        data_matrix = pd.merge(left=data_matrix,
                               right = self.users_features.drop(columns=user_skip_cols),
                               on='user_id',
                               how='left')
        
        item_skip_cols = ['popularity', 'id', 'CURR_SIZE_OF_PRODUCT']
        data_matrix = pd.merge(left=data_matrix,
                               right = self.items_features.drop(columns=item_skip_cols),
                               on='item_id',
                               how='left')

        if type(column_purchases_train) != type(None):
            spam = normalize_data_column(data=data, normalize_column=column_purchases_train)
            spam['flag'] = 1

            data_matrix = pd.merge(left=data_matrix,
                                   right=spam,
                                   on=['user_id', 'item_id'],
                                   how='left')
            
            data_matrix['flag'].fillna(0, inplace=True)
            data_matrix['flag'].astype(int)
        
        return data_matrix

    
    def predict(self, data_test=None, user_id=None, test_preprocess=True):
        if test_preprocess:
            data_test = self.data_test_preprocess(data_test)
            
        rec_cols = ['top_popular', 'top_purchases' , 'top_purchases_by_user', 'own_recommender', 'als_recommender', 'basic_recommender']

        if 'catb_recommender' in self.users_features.columns:
            rec_cols += ['catb_recommender']
                        
        if type(user_id) == type(None):
            data_test_pred = pd.merge(left=data_test,
                                      right=self.users_features[['user_id'] + rec_cols],
                                      how='left',
                                      on='user_id')     
            data_test_pred['hit'] = (~data_test_pred['top_popular'].isnull())*1
            
            data_test_pred['top_popular'] = data_test_pred['user_id'].map(lambda i: self.top_popular[:self.recs_limit_model_lv1])     
            data_test_pred['top_purchases'] = data_test_pred['user_id'].map(lambda i: self.top_purchases[:self.recs_limit_model_lv1])
            
            for col in rec_cols:
                data_test_pred.loc[data_test_pred[col].isnull(), col] = data_test_pred['user_id'].loc[data_test_pred[col].isnull()]\
                                                                        .map(lambda i: self.top_purchases[:self.recs_limit_model_lv1])
            return data_test_pred
        else:
            dict_pred = {'user_id': user_id}
            
            if np.sum(self.users_features['user_id'] == user_id) == 0:
                dict_pred['hit'] = False
                for col in rec_cols:
                    dict_pred[col] = self.users_features.loc[self.users_features['user_id'] == user_id, 'top_popular'].values[0]
            else:
                dict_pred['hit'] = True
                for col in rec_cols:
                    dict_pred[col] = self.users_features.loc[self.users_features['user_id'] == user_id, col].values[0]
                    
            return dict_pred
    
    
    def _add_item_features(self):
        
        self.items_features['dep_assortment'] = self.items_features['DEPARTMENT']\
                                                .map(self.items_features.groupby(by='DEPARTMENT')['item_id'].count().to_dict())
        self.items_features['brand_assortment'] = self.items_features['BRAND']\
                                                  .map(self.items_features.groupby(by='BRAND')['item_id'].count().to_dict())
        self.items_features['commodity_assortment'] = self.items_features['COMMODITY_DESC']\
                                                      .map(self.items_features.groupby(by='COMMODITY_DESC')['item_id'].count().to_dict())
        self.items_features['subcommodity_assortment'] = self.items_features['SUB_COMMODITY_DESC']\
                                                         .map(self.items_features\
                                                              .groupby(by='SUB_COMMODITY_DESC')['item_id'].count().to_dict())
        self.items_features['manufacturar_assortment'] = self.items_features['SUB_COMMODITY_DESC']\
                                                         .map(self.items_features\
                                                              .groupby(by='SUB_COMMODITY_DESC')['item_id'].count().to_dict())         
        self.items_features['popularity'] = self.items_features['item_id']\
                                            .map((self.data.groupby('item_id')['user_id'].nunique() /
                                                  self.data['user_id'].nunique()).to_dict())
        self.items_features['price_avg'] = self.items_features['item_id']\
                                           .map((self.data.groupby('item_id')['sales_value'].sum() /
                                                 self.data.groupby('item_id')['quantity'].sum()).to_dict())
        self.top_popular = self.items_features[(self.items_features['popularity'] <  self.prefilter_params['upper_popularity_limit']) &
                                               (self.items_features['popularity'] > self.prefilter_params['lower_popularity_limit'])]
        self.top_popular = self.top_popular.sort_values(by=['popularity'], ascending=False)['item_id'].to_list()
        
        
    def _add_user_features(self):
        self.users_features = pd.merge(left=pd.DataFrame({'user_id': sorted(pd.concat((self.data['user_id'], 
                                                                                       self.users_features['user_id']),
                                                                                      axis=0).unique())}),
                                       right= self.users_features,
                                       on='user_id',
                                       how='left')
        
        spam = self.data.loc[self.data['week_no'] > (np.max(self.data['week_no']) - self.weeks_model_lv2)]
        purchases_train_dict = spam.groupby(by=['user_id']).agg({'item_id': lambda lst: list(lst)})['item_id'].to_dict()
        self.users_features['purchases_train'] = self.users_features['user_id'].map(purchases_train_dict)
        self.users_features['purchases_train'] = self.users_features['purchases_train'].map(lambda val: val if type(val)==type([]) else [])


    def _prepare_data(self):

        self.data_items = pd.merge(left=self.data,
                                   right=self.items_features[['item_id', 'popularity', 'price_avg', 'dep_assortment']],
                                   how='left',
                                   on='item_id')
        
        period_1 = (np.max(self.data_items['week_no']) - self.weeks_model_lv1-self.weeks_model_lv2)
        period_2 = (np.max(self.data_items['week_no']) - self.weeks_model_lv2)
        self.data_items = self.data_items.loc[(self.data_items['week_no'] > period_1) & (self.data_items['week_no'] <= period_2)]
            
        if 'lower_popularity_limit' in self.prefilter_params.keys():
            self.data_items.loc[self.data_items['popularity'] < self.prefilter_params['lower_popularity_limit'], 'item_id'] = 999999
        if 'upper_popularity_limit' in self.prefilter_params.keys():
            self.data_items.loc[self.data_items['popularity'] > self.prefilter_params['upper_popularity_limit'], 'item_id'] = 999999
        
        if 'lower_price_limit' in self.prefilter_params.keys():
            self.data_items.loc[self.data_items['price_avg'] < self.prefilter_params['lower_price_limit'], 'item_id'] = 999999
        if 'upper_price_limit' in self.prefilter_params.keys():
            self.data_items.loc[self.data_items['price_avg'] > self.prefilter_params['upper_price_limit'], 'item_id'] = 999999
        
        if 'min_dep_assortment' in self.prefilter_params.keys():
            self.data_items.loc[self.data_items['dep_assortment'] < self.prefilter_params['min_dep_assortment'], 'item_id'] = 999999
            
        top_purchases = self.data_items.groupby('item_id')['quantity'].count().reset_index()
        top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.top_purchases = top_purchases[top_purchases['item_id'] != 999999]['item_id'].to_list()

        if 'n_popular_limit' in self.prefilter_params.keys():            
            self.data_items.loc[~self.data_items['item_id']\
                                .isin(self.top_popular[:self.prefilter_params['n_popular_limit']]), 'item_id'] = 999999
            
        self.data_items.drop(columns=['popularity', 'price_avg', 'dep_assortment'], inplace=True)    
        
 
    def _add_top_popular(self):
        self.users_features['top_popular'] = self.users_features['user_id'].map(lambda id: self.top_popular[:self.recs_limit_model_lv1])
        self.users_features['top_popular'] = self.users_features['top_popular'].map(lambda val: val if type(val)==type([]) else [])

    
    def _add_top_purchases(self):
        self.users_features['top_purchases'] = self.users_features['user_id'].map(lambda i: self.top_purchases[:self.recs_limit_model_lv1])
        self.users_features['top_purchases'] = self.users_features['top_purchases'].map(lambda val: val if type(val)==type([]) else [])    
   

    def _add_top_purchases_by_user(self):
        top_purchases_by_user = self.data_items.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        top_purchases_by_user.sort_values(['user_id', 'quantity'], ascending=[True, False], inplace=True)
        top_purchases_by_user = top_purchases_by_user[top_purchases_by_user['item_id'] != 999999]
        top_purchases_by_user = top_purchases_by_user\
                                .groupby(by=['user_id']).agg({'item_id': 
                                                              lambda lst: list(lst)[:self.recs_limit_model_lv1]})['item_id'].to_dict()
        self.users_features['top_purchases_by_user'] = self.users_features['user_id'].map(top_purchases_by_user)
        self.users_features['top_purchases_by_user'] = self.users_features['top_purchases_by_user']\
                                                       .map(lambda val: val if type(val)==type([]) else [])        

        
    def _add_own_recs(self):
        user_item_matrix = (self.user_item_matrix > 0).astype(float)
        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())
        recs = lambda i: [self.id_to_itemid[rec[0]] for rec in own_recommender.recommend(userid=int(i), 
                                                                                         user_items=csr_matrix(user_item_matrix).tocsr(),
                                                                                         N=self.recs_limit_model_lv1,
                                                                                         filter_already_liked_items=False,
                                                                                         filter_items = [self.itemid_to_id[999999]])]
        self.users_features['own_recommender'] = None
        self.users_features.loc[~self.users_features['id'].isnull(), 'own_recommender'] = \
        self.users_features.loc[~self.users_features['id'].isnull(), 'id'].map(recs)
        self.users_features['own_recommender'] = self.users_features['own_recommender'].map(lambda val: val if type(val)==type([]) else [])
 

    def _add_als_recs(self, n_factors=20, regularization=0.001, iterations=20, num_threads=0):
        als_model = AlternatingLeastSquares(factors=n_factors,
                                            regularization=regularization,
                                            iterations=iterations,
                                            num_threads=num_threads)
        
        als_model.fit(csr_matrix(self.user_item_matrix).T.tocsr())
        self.als_model = als_model
        
        als_recs = lambda i: [self.id_to_itemid[rec[0]] for rec in als_model.recommend(userid=int(i), 
                                                                                       user_items=csr_matrix(self.user_item_matrix).tocsr(), 
                                                                                       N=self.recs_limit_model_lv1,
                                                                                       filter_items = [self.itemid_to_id[999999]],
                                                                                       recalculate_user=True,
                                                                                       filter_already_liked_items=False)]
        self.users_features['als_recommender'] = None
        self.users_features.loc[~self.users_features['id'].isnull(), 'als_recommender'] = \
        self.users_features.loc[~self.users_features['id'].isnull(), 'id'].map(als_recs)
        self.users_features['als_recommender'] = self.users_features['als_recommender'].map(lambda val: val if type(val)==type([]) else [])
        
        als_user_factors = pd.DataFrame(self.als_model.user_factors, 
                                        columns=[f'als_user_factor_{i}' for i in range(self.als_model.user_factors.shape[1])])
        als_user_factors['id'] = als_user_factors.index
        self.users_features = pd.merge(left=self.users_features,
                                       right=als_user_factors,
                                       on='id',
                                       how='left')

        als_item_factors = pd.DataFrame(self.als_model.item_factors, 
                                        columns=[f'als_item_factor_{i}' for i in range(self.als_model.item_factors.shape[1])])
        als_item_factors['id'] = als_item_factors.index
        self.items_features = pd.merge(left=self.items_features,
                                       right=als_item_factors,
                                       on='id',
                                       how='left')
    

    def _add_basic_recs(self):
        
        self.users_features['basic_recommender'] = list(zip(self.users_features['top_popular'],
                                                            self.users_features['top_purchases'],
                                                            self.users_features['top_purchases_by_user'],
                                                            self.users_features['als_recommender'],
                                                            self.users_features['own_recommender']))
        self.users_features['basic_recommender'] = self.users_features['basic_recommender']\
                                                   .map(lambda x: np.array(list(zip(x[0], x[1], x[2], x[3], x[4]))).flatten())
    

    def data_test_preprocess(self, data_test):
        data_test_pr = data_test.groupby(by=['user_id']).agg({'item_id': list}).reset_index()
        data_test_pr.rename(columns={'item_id':'purchases'}, inplace=True)

        return data_test_pr


    def _fit_ranking(self):

        cb = catb.CatBoostClassifier(eval_metric='AUC',
                                     silent=True,
                                     iterations=1000,
                                     random_state=21)
        
        data_train = self.data_matrix.copy()
        cat_feat_idx = data_train.dtypes[data_train.dtypes == 'object'].index.to_list()
        data_train[cat_feat_idx] = data_train[cat_feat_idx].fillna('')
        
        cb.fit(data_train.drop(columns=['user_id', 'item_id', 'flag']), data_train['flag'], cat_features=cat_feat_idx)
        self.cb_model = cb
        self.cb_model_columns = data_train.drop(columns=['user_id', 'item_id', 'flag']).columns.to_list()
        
        data_train['proba'] = cb.predict_proba(data_train[self.cb_model_columns])[:,1]
        data_train.drop_duplicates(subset=['user_id', 'item_id'], keep='first', inplace=True)
        data_train.sort_values(by=['user_id', 'proba'], ascending=[True, False], inplace=True)
        data_train['item_id'] = data_train['item_id'].astype(int)
        
        cb_pred_dict = data_train.groupby(by=['user_id']).agg({'item_id': list})['item_id'].to_dict()
        
        self.users_features['catb_recommender'] = self.users_features['user_id'].map(cb_pred_dict)
        