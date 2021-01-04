def prefilter_items(data, item_features, take_n_popular=False):

	from numpy import unique, concatenate

	popularity = data.groupby('item_id')['user_id'].nunique().reset_index() / data['user_id'].nunique()
	popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)

	top_popular = popularity[popularity['share_unique_users'] > 0.5].item_id.tolist()
	data = data[~data['item_id'].isin(top_popular)]

	top_notpopular = popularity[popularity['share_unique_users'] < 0.01].item_id.tolist()
	data = data[~data['item_id'].isin(top_notpopular)]

	sold_12_items = unique(concatenate(data.groupby('week_no')['item_id'].unique()[-52:].values))
	data = data[data['item_id'].isin(sold_12_items)]

	recomended_departments = item_features[item_features['brand'] == 'Private'].\
		groupby('department')['item_id'].nunique().sort_values(ascending=False).index.tolist()

	data = data[(data['sales_value']>1) & (data['sales_value']<200)]

	if take_n_popular:
		popularity_quantity = data.groupby('item_id')['quantity'].sum().reset_index()
		popularity_quantity.rename(columns={'quantity': 'n_sold'}, inplace=True)
		top_n = popularity_quantity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()
		data = data[data['item_id'].isin(top_n)]

	return data



def postfilter_items(user_item_matrix):

	userids = user_item_matrix.index.values
	itemids = user_item_matrix.columns.values

	matrix_userids = np.arange(len(userids))
	matrix_itemids = np.arange(len(itemids))

	id_to_itemid = dict(zip(matrix_itemids, itemids))
	id_to_userid = dict(zip(matrix_userids, userids))

	itemid_to_id = dict(zip(itemids, matrix_itemids))
	userid_to_id = dict(zip(userids, matrix_userids))

	return (id_to_itemid, id_to_userid), (itemid_to_id, userid_to_id)



def get_similar_items_recommendation(user_id, data, itemid_to_id, model, N=5):
	
	def get_rec(model, x):
		recs = model.similar_items(itemid_to_id[x], N=2)
		top_rec = recs[1][0]
		return id_to_itemid[top_rec]


	popularity = data[data['user_id']==user_id].groupby(['item_id'])['quantity'].count().reset_index()
	popularity.sort_values('quantity', ascending=False, inplace=True)
	popularity['similar_recommendation'] = popularity['item_id'].apply(lambda x: get_rec(model, x))

	return popularity['similar_recommendation'].unique()[:N]



def get_similar_users_recommendation(user_id, data, userid_to_id, model, N=5):

	rec_users = model.similar_users(userid_to_id[user_id], N=N*2)
	
	popularity = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
	popularity.sort_values('quantity', ascending=False, inplace=True)
	popularity = popularity.groupby('user_id').head(N)
	popularity.sort_values('user_id', ascending=False, inplace=True)
	
	id_to_emb = {user_id: emb for user_id, emb in rec_users}
	popularity = popularity[popularity['user_id'].isin(id_to_emb.keys())]
	
	popularity['similar_recommendation'] = popularity['user_id'].apply(lambda x: id_to_emb[x]) * popularity['quantity']
	popularity.sort_values('similar_recommendation', ascending=False, inplace=True)
	
	return popularity['item_id'].head(N).values
