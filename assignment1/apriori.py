import pandas as pd
import numpy as np
from itertools import chain, combinations
import sys

def getItemsetFromTdb(tdb: list):
    # Get all single element sets from transaction DB
        # tdb: list - list of transaction(type of set)
        # return: list - single element itemsets

    return list(map(lambda x: {x}, (set([i for t in tdb for i in t]))))


def getSupport(tdb: list, item: set):
    # Get support of item passed by argument
        # tdb: list - list of transaction(type of set)
        # return support by percentage
    
    _sup = 0
    for t in tdb:
        if item.issubset(t):
            _sup += 1

    return _sup / len(tdb)


def getCandidate(itemset: list, length: int):
    # Get next-step candidates from frequent itemsets
        # itemset: list - list of set type items (C(k))
        # length: int - length of set type items of current itemsets
        # return: list - C(k + 1)

    _ck = []

    if length == 1:
        [_ck.append(i.union(j)) for i in itemset for j in itemset if i != j]
    else: 
        for (i, j) in combinations(itemset, 2):
            _sd = i.symmetric_difference(j)
            if len(_sd) == length and _sd in itemset:
                _ck.append(i.union(j))

    result = []
    [result.append(i) for i in _ck if i not in result]

    return result


def filterBySup(tdb: list, itemset: list, sup: float):
    # filtering candidate by its support
        # tdb: list - list of transaction(type of set)
        # itemset: list - list of itemsets
        # sup: int - percentage of minimun support
        # return: list - frequent itemsets (L(k))

    return [i for i in itemset if getSupport(tdb, i) >= sup]


def getFrequentItemset(tdb: list, sup: float):
    # get Frequent itemsets from transaction DB
        # tdb: list - list of transaction (type of set)
        # sup: float - percentage of minimun support
        # return: list - list of frequent itemsets

    _Ck = getItemsetFromTdb(tdb)
    _Lk = filterBySup(tdb, _Ck, sup)
    _L = []
    _length = 1

    while len(_Lk) != 0:
        _L.extend(_Lk)
        _Ck = getCandidate(_Lk, _length)
        _Lk = filterBySup(tdb, _Ck, sup)
        _length += 1

    return _L

def getSubsets(itemset: set):
    # get subsets of set except null set
        # itemset: set - itemset
        # return: list - list of subsets

    _ssl = []
    _item_chain = chain([combinations(itemset, i+1) for i in range(len(itemset))])
    [_ssl.append(set(i)) for c in _item_chain for i in c ]

    return _ssl


def apriori(tdb: list, sup: float):
    # get frequent itemsets and its support from transaction DB with minimum support
        # tdb: list - list of transaction (type of set)
        # sup: float - percentage of minimun support
        # return: dataframe(columns=[itemset, support]) - frequent itemsets and its support

    _freq_itemsets = getFrequentItemset(tdb, sup)
    
    _df = pd.DataFrame(data=pd.Series(_freq_itemsets), columns=['itemset'])
    _df['support'] = _df['itemset'].map(lambda x: getSupport(tdb, x))

    return _df


def association_rule(frequent_itemsets):
    # get association and its confidence from frequent itemsets
        # frequent_itemsets: dataframe - frequent itemsets and its support
        # return: dataframe(columns=[antecedent, consequent, support, confidence]) 

    _ar = []

    for row in frequent_itemsets.iterrows():
        _itemset = row[1]['itemset']
        if len(_itemset) == 1:
            continue

        _subsets = getSubsets(_itemset)
        for _ant in _subsets:
            _cons = _itemset.difference(_ant)
            if len(_cons) > 0:
                _union_sup = frequent_itemsets[frequent_itemsets['itemset'] == _itemset].reset_index().loc[0, 'support']
                _ant_sup = frequent_itemsets[frequent_itemsets['itemset'] == _ant].reset_index().loc[0, 'support']
                _conf = _union_sup / _ant_sup
                _ar.append((_ant, _cons, round(_union_sup * 100, 2), round(_conf * 100, 2)))

    return pd.DataFrame(_ar, columns = ['antecedent', 'consequent', 'support', 'confidence'])
    

def main(sup, input_file, output_file):
    # main function
        # sup: str - minimum support from command line
        # input_file: str - name of input file
        # output_file: str - name of output file
        
    # file  open
    f = open(input_file, 'r')
    lines = f.readlines()
    f.close()
    
    # make transaction list from input.txt
    tdb = []
    for l in lines:
        tdb.append(set(l.split()))

    # running algorithm
    ap = apriori(tdb, int(sup) / 100)
    ar = association_rule(ap)

    # make ouput.txt
    f = open(output_file, 'w')
    for row in ar.iterrows():
        d  = row[1].to_dict()
        f.write('{{{0}}}\t{{{1}}}\t{2:.2f}\t{3:.2f}\n'. format(','.join(d['antecedent']), ','.join(d['consequent']), d['support'], d['confidence']))
    f.close()


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
