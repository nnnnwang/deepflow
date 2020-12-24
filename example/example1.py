# -*- coding: utf-8 -*-
# @Time    : 2020/12/5
# @Author  : 爱喝可乐的学渣
# @Email   : wangnannanzju@qq.com

import pandas as pd
import numpy as np

from common.const import Const
from model import DeepFM,DNN,FeatureSelector
from preprocess import FeatureProcess

# 生成原始特征
df_dict={'f1':[np.random.rand() for _ in range(10)],
         'f2':['a'+str(np.random.random_integers(0,10))+'^b' for _ in range(10)]}
label = np.random.randint(0, 2,size=(10,1))
df = pd.DataFrame(df_dict)
df['f2'] = df['f2'].apply(lambda x:x.split(Const.SEQUENCE_DELIMITER))
print(df['f2'].head(10))

'''
这之间做特征工程
'''

# 特征预处理，连续特征归一化，离散特征编码
fp = FeatureProcess()
feature_cls_list, input_data = fp.feature_process(df, ['f1'], ['f2'])

model = FeatureSelector(feature_cls_list)
model.desc()

model.compile('sgd', loss='binary_crossentropy', metrics=['acc'])

model.fit(input_data, label, epochs=10, steps_per_epoch=1, callbacks=[model.get_feature_importance_cb()])
