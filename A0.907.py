#环境
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')


#1.2 数据读取
train_data = pd.read_csv('data/dataTrain.csv',skipinitialspace=True)
test_data = pd.read_csv('data/dataA.csv')
submission = pd.read_csv('data/submit_example_A.csv')
data_nolabel = pd.read_csv('data/dataNoLabel.csv')

print(f'train_data.shape = {train_data.shape}')
print(f'test_data.shape  = {test_data.shape}')
# print([column for column in train_data])
# 1.3 特征构造
# 自己DIY的特征
# print(train_data)
# print(train_data['id'])
train_data['f47'] = train_data["f1"] * 10 + train_data['f2']
test_data['f47'] = test_data['f1'] * 10 + test_data['f2']

# 暴力Feature 位置
loc_f = ['f1', 'f2', 'f4', 'f5', 'f6']
for df in [train_data, test_data]:
    # print(df)
    for i in range(len(loc_f)):
        for j in range(i + 1, len(loc_f)):
            df[f'{loc_f[i]}+{loc_f[j]}'] = df[loc_f[i]] + df[loc_f[j]]
            df[f'{loc_f[i]}-{loc_f[j]}'] = df[loc_f[i]] - df[loc_f[j]]
            df[f'{loc_f[i]}*{loc_f[j]}'] = df[loc_f[i]] * df[loc_f[j]]
            df[f'{loc_f[i]}/{loc_f[j]}'] = df[loc_f[i]] / (df[loc_f[j]]+1)
# print(train_data)
# 暴力Feature 通话
com_f = ['f43', 'f44', 'f45', 'f46']
for df in [train_data, test_data]:
    
    for i in range(len(com_f)):
        for j in range(i + 1, len(com_f)):
            df[f'{com_f[i]}+{com_f[j]}'] = df[com_f[i]] + df[com_f[j]]
            df[f'{com_f[i]}-{com_f[j]}'] = df[com_f[i]] - df[com_f[j]]
            df[f'{com_f[i]}*{com_f[j]}'] = df[com_f[i]] * df[com_f[j]]
            df[f'{com_f[i]}/{com_f[j]}'] = df[com_f[i]] / (df[com_f[j]]+1)


#ID类特征数值化
cat_columns = ['f3']
data = pd.concat([train_data, test_data])

for col in cat_columns:
    lb = LabelEncoder()
    lb.fit(data[col])
    train_data[col] = lb.transform(train_data[col])
    test_data[col] = lb.transform(test_data[col])

#最后构造出训练集和测试集
num_columns = [ col for col in train_data.columns if col not in ['id', 'label', 'f3']]
feature_columns = num_columns + cat_columns
target = 'label'

train = train_data[feature_columns]
label = train_data[target]
print(label)
test = test_data[feature_columns]
#1.4 模型训练代码
def model_train(model, model_name, kfold=5):
    oof_preds = np.zeros((train.shape[0]))
    test_preds = np.zeros(test.shape[0])
    skf = StratifiedKFold(n_splits=kfold)#均匀的分割数据集，保持和原本数据一样的样本比例
    print(f"Model = {model_name}")
    for k, (train_index, test_index) in enumerate(skf.split(train, label)):
        x_train, x_test = train.iloc[train_index, :], train.iloc[test_index, :]
        y_train, y_test = label.iloc[train_index], label.iloc[test_index]

        model.fit(x_train,y_train)

        y_pred = model.predict_proba(x_test)[:,1]
        oof_preds[test_index] = y_pred.ravel()#将数组拉成一维数组
        auc = roc_auc_score(y_test,y_pred)
        print("- KFold = %d, val_auc = %.4f" % (k, auc))
        test_fold_preds = model.predict_proba(test)[:, 1]
        test_preds += test_fold_preds.ravel()
        if k==3:
            return test_preds /4
    print("Overall Model = %s, AUC = %.4f" % (model_name, roc_auc_score(label, oof_preds)))
    return test_preds / kfold
gbc = GradientBoostingClassifier()
gbc_test_preds = model_train(gbc, "GradientBoostingClassifier", 5)


submission['label'] = gbc_test_preds
submission.to_csv('submission.csv', index=False)
