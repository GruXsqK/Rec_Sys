import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix

from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    
    def __init__(self, data, weighting=True):
                
        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = prepare_dicts(self.user_item_matrix)
        
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T 
        
        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
     
    @staticmethod
    def prepare_matrix(data):
        
        user_item_matrix = pd.pivot_table(data,
                                          index='user_id',
                                          columns='item_id',
                                          values='quantity',
                                          aggfunc='count',
                                          fill_value=0).astype(float)
        
        return user_item_matrix
    
    @staticmethod
    def prepare_dicts(user_item_matrix):
        
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
        
        model = AlternatingLeastSquares(factors=factors, 
                                             regularization=regularization,
                                             iterations=iterations,  
                                             num_threads=num_threads)
        model.fit(csr_matrix(self.user_item_matrix).T.tocsr())
        
        return model

    def get_similar_items_recommendation(self, user_id, data, itemid_to_id, model, N=5):

        def _get_rec(model, x):
            recs = model.similar_items(itemid_to_id[x], N=2)
            top_rec = recs[1][0]
            return id_to_itemid[top_rec]


        popularity = self.data[self.data['user_id']==self.user_id].groupby(['item_id'])['quantity'].count().reset_index()
        popularity.sort_values('quantity', ascending=False, inplace=True)
        popularity['similar_recommendation'] = popularity['item_id'].apply(lambda x: _get_rec(self.model, x))
        res = popularity['similar_recommendation'].unique()[:self.N]

        assert len(res) == self.N, 'Количество рекомендаций != {}'.format(self.N)
        return res
    
    def get_similar_users_recommendation(self, user_id, data, userid_to_id, model, N=5):
    	
        rec_users = self.model.similar_users(self.userid_to_id[self.user_id], N=self.N*2)

        popularity = self.data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        popularity.sort_values('quantity', ascending=False, inplace=True)
        popularity = popularity.groupby('user_id').head(self.N)
        popularity.sort_values('user_id', ascending=False, inplace=True)

        id_to_emb = {user_id: emb for user_id, emb in rec_users}
        popularity = popularity[popularity['user_id'].isin(id_to_emb.keys())]

        popularity['similar_recommendation'] = popularity['user_id'].apply(lambda x: id_to_emb[x]) * popularity['quantity']
        popularity.sort_values('similar_recommendation', ascending=False, inplace=True)

        res = popularity['item_id'].head(self.N).values
        
        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
