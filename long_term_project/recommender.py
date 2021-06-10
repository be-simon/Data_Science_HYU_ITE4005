import sys
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_sparse_matrix(user_item_list):    
    # ---------- 
    # user-item-rating list를 sparse matrix로 변환해준다.
    #   user_item_list : user-item-rating을 row로 갖는 dataframe
    #   return : user-item을 행, 열로 갖고 rating을 나타내는 sparse matrix
    # ----------

    _list_columns = user_item_list.columns
    _index, _columns, _value = _list_columns

    _sparse = user_item_list.pivot_table(_value,index=_index, columns=_columns).fillna(0)
    
    return _sparse
    

def cal_item_similarity(user_item_rating):
    # ---------- 
    # item 간의 코사인 유사도를 구한다.
    #   user_item_rating : user-item-rating을 나타내는 sparse matrix
    #   return : item 간의 코사인 유사도를 계산한 array
    # ----------
    
    return cosine_similarity(user_item_rating.T)

def get_user_rating_avg(user_item_rating, user_id):
    # ---------- 
    # user가 item들에 대해 매긴 평점의 평균을 구한다.    
    #   user_item_rating : user-item-rating을 나타내는 sparse matrix
    #   user_id : 구하고자 하는 유저의 id
    #   return : 유저가 매긴 평점들의 평균
    # ----------

    user_rating = user_item_rating.loc[user_id]
    user_rating_drop = user_rating[user_rating > 0]

    return int(round(np.average(user_rating_drop.values), 0))


def expect_user_item_rating(user_item_rating, sim_array, user_id, item_id):
    # ---------- 
    # user가 item에 대해 매긴 평점을 예측한다.
    #   user_item_rating : user-item-rating을 나타내는 sparse matrix
    #   sim_array : 아이템 간의 코사인 유사도
    #   user_id : 구하고자 하는 유저의 id
    #   item_id : 예측하고자 하는 아이템의 id
    #   return : 아이템에 대한 평점의 예측값 (정수)
    # ----------

    # 유저가 아이템들에 매긴 평점과, 구하려는 아이템과 다른 아이템들간의 유사도를 가져온다.
    user_rating = user_item_rating.loc[user_id]
    item_index = np.where(user_item_rating.columns == item_id)[0][0]
    item_corr = sim_array[item_index]

    rating_corr =  np.array([user_rating, item_corr]).T
    rating_corr_drop = rating_corr[rating_corr[:, 0] > 0] # 유저가 평점을 매기지 않은 아이템들은 지운다.
    top_rating = rating_corr_drop[np.argsort(rating_corr_drop, axis=0).T[1]][::-1][0:13, :].T # 유사도 상위 13개 아이템을 가져온다.
    
    return int(round(np.average(top_rating[0],weights=top_rating[1]), 0))


if __name__ == '__main__':
    base_file_name = sys.argv[1]
    test_file_name = sys.argv[2]

    columns = ['user_id', 'item_id', 'rating', 'time_stamp']
    base_file = pd.read_csv('./data/u1.base', sep='\t', names=columns)
    test_file = pd.read_csv('./data/u1.test', sep='\t', names=columns)

    user_item_list = base_file.drop('time_stamp', axis=1)    
    test_rating = test_file['rating'].values
    test_user_item_list = test_file.drop(['time_stamp', 'rating'], axis=1)

    user_item_rating = get_sparse_matrix(user_item_list)
    sim_array = cal_item_similarity(user_item_rating)


    result = []
    # 각 user-item 쌍에 대해 예측값을 구한다.
    # 기존 데이터에 없던 item에 대해서는 유저가 매긴 평점들의 평균으로 예측한다.
    for idx, user_id, item_id in test_user_item_list.itertuples():
        if item_id in user_item_rating.columns:
            result.append(expect_user_item_rating(user_item_rating, sim_array, user_id, item_id))
        else:
            result.append(get_user_rating_avg(user_item_rating, user_id))

    
    
    
    test_user_item_list['rating'] = result
    test_user_item_list.to_csv(f'{base_file_name}_prediction.txt', sep='\t', index=False, header=False)
