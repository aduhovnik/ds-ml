# coding: utf-8

# In[2]:


import warnings
import pickle

warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# In[3]:


PATH_TO_DATA = '/home/andrei/Desktop/alice_kaggle'
train_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_sessions.csv'), index_col='session_id')
test_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'test_sessions.csv'), index_col='session_id')
train_df.sort_values('time1')
# train_df = train_df.loc[:10000]
# test_df = test_df.loc[:10000]
y = train_df['target']
train_df.drop('target', axis=1, inplace=True)
times = ['time%s' % i for i in range(1, 11)]
train_df[times] = train_df[times].apply(pd.to_datetime)
test_df[times] = test_df[times].apply(pd.to_datetime)

# filter here by dd[(dd['time1'] > '2013-09-01')]['time1'].hist()

train_df.head()

# In[15]:


dd = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_sessions.csv'), index_col='session_id')
dd['time1'] = dd['time1'].apply(pd.to_datetime)

# In[139]:


print(len(dd))
print(len(dd[dd['target'] == 1]))
print(len(dd[dd['target'] == 0]))

print(dd[:10]['time1'].dt.weekday_name)
print(dd[:10]['time1'].dt.dayofweek)

# In[135]:


import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker


def show_heatmap_dayofweek_hour(df):
    df['dayofweek'] = df['time1'].dt.dayofweek
    df['hour'] = df['time1'].dt.hour

    df['count'] = df.groupby(['dayofweek', 'hour'])['target'].transform("count")
    df = df.drop_duplicates(subset=['dayofweek', 'hour'])

    df = resdf.merge(df, on=['dayofweek', 'hour'])

    df = df.pivot('dayofweek', 'hour', 'count_y')
    df = df.fillna(0)

    f, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(df, annot=True, fmt="d")


show_heatmap_dayofweek_hour(dd[dd['target'] == 1])

# In[131]:


show_heatmap_dayofweek_hour(dd[dd['target'] == 0])

# In[143]:


dd['hour'] = dd['time1'].dt.hour
dd[dd['target'] == 0]['hour'].hist()

# In[144]:


dd['hour'] = dd['time1'].dt.hour
dd[dd['target'] == 1]['hour'].hist()

# In[4]:


# загрузим словарик сайтов
with open(r"/home/andrei/Desktop/alice_kaggle/site_dic.pkl", "rb") as input_file:
    site_dict = pickle.load(input_file)

# датафрейм словарика сайтов
sites_dict_df = pd.DataFrame(list(site_dict.keys()),
                             index=list(site_dict.values()),
                             columns=['site'])
print(u'всего сайтов:', sites_dict_df.shape[0])

text_columns = ['site_column%s' % i for i in range(1, 11)]
train_df['text_col'] = ''
test_df['text_col'] = ''

for i in range(1, 11):
    site_c = 'site{}'.format(i)
    site_name_c = 'site_column{}'.format(i)
    train_df[site_name_c] = sites_dict_df.loc[train_df[site_c]].values
    test_df[site_name_c] = sites_dict_df.loc[test_df[site_c]].values
    train_df['text_col'] += train_df[site_name_c]
    test_df['text_col'] += test_df[site_name_c]

print(train_df.head())

all_train_text = pd.concat([train_df['site_column{}'.format(i)].astype('U') for i in range(1, 11)])
char_vec = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), max_features=100000)
word_vec = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), max_features=100000)
# fix wordofvec
word_vec.fit(all_train_text.astype('U'))

# In[5]:


X_tfidf = word_vec.transform(pd.concat((train_df['text_col'], test_df['text_col'])).values.astype('U'))

# In[13]:


alice_common = dict()
for i in range(1, 11):
    a = train_df[y == 0]['site_column%s' % i].value_counts()
    for k, v in dict(a).items():
        alice_common.setdefault(k, 0)
        alice_common[k] += v
alice_common = list(alice_common.items())
alice_common.sort(key=lambda x: x[1], reverse=True)
alice_common = set([_[0] for _ in alice_common[:10]])


def get_part_from_hour(h):
    return h // 3


def get_session_duration(row):
    time_values = row[['time%s' % i for i in range(1, 10)]].dropna().values
    duration = (time_values.max() - time_values.min()).total_seconds()
    return duration


def get_count_of_good_sites(row):
    res = 0
    for i in range(1, 11):
        res += row['site_column%s' % i] in alice_common
    return res


def add_extra_features(df):
    df['hour'] = df['time1'].dt.hour

    for i in range(2, 4):
        df['delta%s' % (i - 1)] = (df['time%s' % i] - df['time%s' % (i - 1)]).dt.total_seconds()

    # df['duration'] = df.apply(get_session_duration, axis=1)

    df['count_good'] = df.apply(get_count_of_good_sites, axis=1)

    return df


def get_scaled_features(df):
    scalable_columns = ['hour', 'delta1', 'count_good']  # 'duration']
    return MinMaxScaler().fit_transform(df[scalable_columns].fillna(-1))


def get_OH(df):
    df['part_of_day'] = np.array(list(map(lambda v: get_part_from_hour(v), df['time1'].dt.hour)))

    df['hour'] = df['time1'].dt.hour
    df['is_work_time'] = df['hour'].apply(lambda x: 8 <= x <= 17)

    df['dayofweek'] = df['time1'].dt.dayofweek

    return OneHotEncoder().fit_transform(df[['part_of_day', 'is_work_time', 'dayofweek', ]])


def get_extra_features(df):
    return get_scaled_features(add_extra_features(df))


# In[6]:


dir(train_df['time1'].dt)

# In[11]:


full_df = pd.concat([train_df, test_df])
sites = ['site%s' % i for i in range(1, 11)]
# Index to split the training and test data sets
idx_split = train_df.shape[0]
full_sites = full_df[sites]
full_sites.head()

sites_flatten = full_sites.values.flatten()

# and the matrix we are looking for
full_sites_sparse = csr_matrix(([1] * sites_flatten.shape[0],
                                sites_flatten,
                                range(0, sites_flatten.shape[0] + 10, 10)))[:, 1:]

# In[67]:


get_ipython().run_cell_magic('time', '',
                             "_train = pd.concat((train_df, test_df))\nX_train = hstack([#full_sites_sparse, X_tfidf,\n                  get_OH(_train), \n                  get_extra_features(_train)\n                 ], format='csr')")


# In[19]:


def get_auc_lr_valid(X, y, C=0.07, seed=17, ratio=0.7):
    idx = int(round(X.shape[0] * ratio))
    lr = LogisticRegression(C=C, random_state=seed, n_jobs=-1).fit(X[:idx, :], y[:idx])
    y_pred = lr.predict_proba(X[idx:, :])[:, 1]
    score = roc_auc_score(y[idx:], y_pred)

    return score


# In[36]:


tscv = TimeSeriesSplit(n_splits=3)
lr = LogisticRegression(random_state=17, n_jobs=-1)
param_search = {'C': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]}
lrg = GridSearchCV(estimator=lr, cv=tscv,
                   param_grid=param_search)
lrg.fit(X_train[:idx_split], y)

# In[37]:


dir(lrg)
print(lrg.best_params_)
print(lrg.cv_results_)
print(lrg.best_score_)

# In[68]:


get_ipython().run_cell_magic('time', '', 'print(get_auc_lr_valid(X_train[:idx_split], y, C=5))')

# In[94]:


from xgboost import XGBClassifier


def get_auc_xgboost_valid(X, y, seed=17, ratio=0.4):
    idx = int(round(X.shape[0] * ratio))
    lr = XGBClassifier(random_state=seed, n_jobs=-1, scorring='roc_auc', max_depth=6).fit(X[:idx, :], y[:idx])
    y_pred = lr.predict_proba(X[idx:, :])[:, 1]
    score = roc_auc_score(y[idx:], y_pred)

    return score


# In[95]:


get_ipython().run_cell_magic('time', '', 'print(get_auc_xgboost_valid(X_train[:idx_split], y))')


# In[43]:


# Function for writing predictions to a file
def write_to_submission_file(predicted_labels, out_file,
                             target='target', index_label="session_id"):
    predicted_df = pd.DataFrame(predicted_labels,
                                index=np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)


# In[60]:


get_ipython().run_cell_magic('time', '',
                             "#lr = LogisticRegression(C=5, random_state=17, n_jobs=-1).fit(X_train[:idx_split], y)\nlr = XGBClassifier(random_state=17, n_jobs=-1, scorring='roc_auc').fit(X_train[:idx_split], y)\nX_test = X_train[idx_split:,:]\ny_test = lr.predict_proba(X_test)[:, 1]\n\n# Write it to the file which could be submitted\nwrite_to_submission_file(y_test, 'alice.csv')")
