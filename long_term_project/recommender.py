import sys
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_sparse_matrix(user_item_list):    
    _list_columns = user_item_list.columns
    _index, _columns, _value = _list_columns

    _sparse = user_item_list.pivot_table(_value,index=_index, columns=_columns).fillna(0)
    
    return _sparse
    

def cal_item_similarity(user_item_rating):
    # cosine similarity
    sim_array = cosine_similarity(user_item_rating.T)

    # item - item similarity
    item_index = user_item_rating.columns
    return pd.DataFrame(sim_array, index=item_index, columns=item_index)
    

def expect_user_item_rating(user_item_rating, sim_df, user_id, item_id):
    user_rating = user_item_rating.loc[user_id]
    item_corr = sim_df[item_id]
    

    rating_corr = pd.DataFrame([user_rating, item_corr], index=['rating', 'correlation']).T
    top_rating = rating_corr[rating_corr['rating'] > 0].sort_values('correlation', ascending=False).head(10)
    
    values = top_rating['rating'].values
    weight = top_rating['correlation'].values
    
    return np.dot(values, weight) / sum(weight)



if __name__ == '__main__':
    base_file_name = sys.argv[1]
    test_file_name = sys.argv[2]

    columns = ['user_id', 'item_id', 'rating', 'time_stamp']
    base_file = pd.read_csv('./data/u1.base', sep='\t', names=columns)
    test_file = pd.read_csv('./data/u1.test', sep='\t', names=columns)

    user_item_list = base_file.drop('time_stamp', axis=1)    
    test_user_item_list = test_file.drop(['time_stamp', 'rating'], axis=1)

    user_item_rating = get_sparse_matrix(user_item_list)
    sim_df = cal_item_similarity(user_item_rating)


    result = []
    for idx, user_id, item_id in test_user_item_list.itertuples():
        if item_id in user_item_rating.columns:
            result.append(expect_user_item_rating(user_id, item_id))
        else:
            result.append(0)

    test_user_item_list['rating'] = result
    test_user_item_list.to_csv(f'{base_file_name}_prediction.txt', sep='\t', index=False, header=False)
