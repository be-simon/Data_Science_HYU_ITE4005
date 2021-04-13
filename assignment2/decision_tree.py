import pandas as pd
import numpy as np
import sys
import random

def get_entropy(data):
    # 
    # 주어진 data의 entropy를 계산
        # return (float): data's entropy
    #
    key, counts = np.unique(data, return_counts=True)
    return -np.sum([counts[i] / np.sum(counts) * np.log2(counts[i] / np.sum(counts)) for i in range(len(key))])


def get_info_gain(data, attr, target):
    #
    # attr의 information gain을 계산
        # data (DataFrame): 전체 data
        # attr (str): information gain을 계산할 attribute
        # target (str): 분류의 대상이 되는 class
        # return (float): information gain of attribute
    #   

    # attribute 전체의 entropy
    total_entropy = get_entropy(data[target])
    
    # attribute 값들의 entropy를 구하고 가중 평균을 구한다.
    labels, counts = np.unique(data[attr], return_counts=True)
    attr_entropy = np.sum([cnt * (get_entropy(data[data[attr] == l][target])) for l, cnt in zip(labels, counts)]) / np.sum(counts)
    
    return total_entropy - attr_entropy

def get_next_attr(data, target):
    #
    # information gain을 비교해서 다음 기준 attribute를 선정
        # data (DataFrame): data for classification
        # return (str): next attribute name
    #

    attrs = list(data.columns)[:-1]
    index = np.array([get_info_gain(data, attr, target) for attr in attrs]).argmax()

    return attrs[index]

def decision_tree(data, target):
    #
    # 주어진 data로부터 decision tree model을 생성
        # data (DataFrame): train data
        # target (str): 분류의 대상이 되는 class
        # return (dict): node of decision tree
    #

    # 남아있는 data들의 class 값이 일치할 때
    if len(np.unique(data[target])) <= 1:
        return np.unique(data[target])[0]

    # 비교할 속성이 없을 때
    elif len(data.columns) <= 1: # class column만 남음
        key, counts = np.unique(data[target], return_counts=True)
        return key[counts.argmax()] # majority voting 방식으로 class 값 리턴

    else:
        # 다음 attribute를 선정하고 dict type으로 node를 만든다
        next_attr = get_next_attr(data, target)
        node = {next_attr: {}}
        labels = np.unique(data[next_attr])
        # attribute의 값에 따라 tree 연장
        for l in labels:
            sub_data = data[data[next_attr] == l].drop([next_attr], axis=1)
            node[next_attr][l] = decision_tree(sub_data, target)

        return node

def test_decision_tree(decision_tree, test_file, target):
    #
    # 주어진 test 파일을 decision tree model으로 분류
        # decision_tree (dict): tree model
        # test_file (DataFrame): classify test file
        # target (str): class column name (classification answer)
        # return (DataFrame): test file's classification answer
    #

    answer = []
    for i in range(len(test_file.index)):
        subtree = decision_tree
        # class value (str)을 만날 때까지 반복
        while type(subtree) == type({}):
            attr = list(subtree.keys())[0]
            label = test_file.loc[i, attr]
            
            if label in subtree[attr]: # 트리에 있는 값
                subtree = subtree[attr][label]
            else: 
                # 트리에 없는 값을 만났을 때
                # class가 결정되는 attr과 노드가 이어지는 attr을 구한다.
                values = list(subtree[attr].values())
                target_value = [v for v in values if type(v) == type('')]
                next_node = [n for n in values if type(n) == type({})]
                
                if len(target_value) > 0: # class가 결정되는 값이 있다면 그 중 더 많은 값으로 결정
                    key, counts = np.unique(target_value, return_counts=True)
                    subtree = key[counts.argmax()]
                else: # 아니라면 다음 노드로
                    subtree = next_node[0]
                
        answer.append(subtree)

    test_file[target] = answer
    
    return test_file

if __name__ == '__main__':
    # command line arguments
    train_file_name = sys.argv[1]
    test_file_name = sys.argv[2]
    result_file_name = sys.argv[3]

    # open files
    train_file = pd.read_csv(train_file_name, sep='\t')
    test_file = pd.read_csv(test_file_name, sep='\t')
    target = train_file.columns[-1]

    # generate tree model & test    
    decision_tree = decision_tree(train_file, target)
    result = test_decision_tree(decision_tree, test_file, target)

    result.to_csv(result_file_name, sep='\t', index=False)
    
