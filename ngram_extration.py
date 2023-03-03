# -*- coding:utf-8 -*-
"""
-------------------------------------------------
   File Name：     ngram_extration
   Author :        Biao Liu
   Create Data：    2023/3/3 11:05
   Description :
-------------------------------------------------
"""

# 挖掘数据中的脏字符
# 利用高频n-gram挖掘藏字符
# 通过迭代将短n-gram融合到长n-gram，得到全长的脏字符串
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer

def merge(p_list, s_list, feature_fre_dict):
    # 融合父特征和子特征
    # 如果子特征词频高于父特征，子特征取代父特征，否则不取代；
    '''
    p_list: ['abcd', 'bcde']
    s_list: ['abc', 'bcd', 'cde']
    feature_fre_dict:
    {
        'abcd': 4,
        'bcde': 4,
        'abc': 4,
        'bcd': '8',
        'cde': 4
    }
    merge(['abcd', 'bcde'], ['abc', 'bcd', 'cde'], {'abcd': 4, 'bcde': 4, 'abc': 4, 'bcd': '8', 'cde': 4})
    return:
    ['bcd']
    '''
    p_list_bp = p_list[:]
    feature_remove_set = set()
    for p in p_list:
        for s in s_list:
            if s in p:
                feature_remove_set.add(s)
                if feature_fre_dict[s] == feature_fre_dict[p]:
                    continue
                else:
                    try:
                        p_list_bp[p_list_bp.index(p)] = s
                    except:
                        p_list_bp.append(s)
    add_list = [w for w in s_list if w not in feature_remove_set]
    return list(dict.fromkeys(p_list_bp + add_list))  # 去除冗余并保持顺序不变

def reduce(feature_list):
    # 内部融合
    '''
    即比较最终的特征list，融合父特征和子特征，即如果存在子特征和父特征关系，就去掉该子特征；
    因为经过merge融合，最终剩余的特征中，如果存在父子特征对，那么父特征词频必然等于子特征词频，因此可以不比较词频，直接删除子特征；
    reduce(['abcd', 'abc', 'ab'])
    return: ['abcd']
    '''
    feature_list = sorted(feature_list, key=lambda k:len(k), reverse=True)
    remove_set = set()
    for i in range(len(feature_list) - 1):
        p = feature_list[i]
        for j in range(i+1, len(feature_list)):
            s = feature_list[j]
            if s in p:
                remove_set.add(s)
    return [w for w in feature_list if w not in remove_set]

class NgramExtration:
    def __init__(self):
        self.vec = None
        self.vector = None
        self.feature_list = None
        self.feature_fre_dict = None
    
    def extract_ngram(self, contents, min_df=0.001, ngram_range=(5,30), analyzer='char', lowercase=False):
        '''
        作用：从contents中提取高频的最大词频情况下的最长ngram字符串；
        应用场景：挖掘语料库中的脏字符串用于数据预处理（挖掘的是高频词，因此min_df设置高一些，如0.001或者20），
        挖掘候选的特征词用于特征标注或者特征选择（挖掘的是一般的词，因此min_df设置小一些，如5）；
        '''
        # 第一步：得到所有的n-gram矩阵
        self.vec = CountVectorizer(min_df=min_df, ngram_range=ngram_range, analyzer=analyzer, lowercase=lowercase)
        self.vector = self.vec.fit_transform(tqdm(contents))
        # 第二步：得到每个ngram的词频，并不断将短ngram融合到长ngram
        fre = np.array(self.vector.sum(axis=0))[0]
        self.feature_list = self.vec.get_feature_names()
        self.feature_fre_dict = {self.feature_list[i]: fre[i] for i in range(len(self.feature_list))}
        feature_list_sorted = sorted(self.feature_list, key=lambda k:len(k), reverse=True)
        
        # 将features按照长度进行分层存储
        feature_list_by_length = []  # 分层存储
        length = len(feature_list_sorted[0])
        class_list = []
        for w in feature_list_sorted:
            if len(w) < length:
                feature_list_by_length.append(class_list)
                length = len(w)
                class_list = [w]
            else:
                class_list.append(w)
        feature_list_by_length.append(class_list)
        
        # 分层迭代融合
        parent_feature = feature_list_by_length[0]
        # print(parent_feature)
        for features in tqdm(feature_list_by_length[1:]):
            parent_feature = merge(parent_feature, features, self.feature_fre_dict)
            # print(parent_feature)
        # 最终内部融合
        parent_feature = reduce(parent_feature)
        final_feature_list = [[f, self.feature_fre_dict[f]] for f in parent_feature]
        return final_feature_list