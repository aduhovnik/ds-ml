# coding: utf-8

# <center>
# <img src="../../img/ods_stickers.jpg">
# ## Open Machine Learning Course
# <center>
# Author: Yury Kashnitsky, Data Scientist at Mail.Ru Group
# 
# This material is subject to the terms and conditions of the license [Creative Commons CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Free use is permitted for any non-comercial purpose with an obligatory indication of the names of the authors and of the source.

# ## <center>Assignment #6. Part 1
# ### <center> Beating benchmarks in "Catch Me If You Can: Intruder Detection through Webpage Session Tracking"
#     
# [Competition](https://www.kaggle.com/c/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2). The task is to beat "Assignment 6 baseline".

# In[11]:


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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Reading original data

# In[175]:


PATH_TO_DATA = '/home/andrei/Desktop/alice_kaggle'
train_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_sessions.csv'), index_col='session_id')
test_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'test_sessions.csv'), index_col='session_id')
# train_df = train_df.loc[:10000]
# test_df = test_df.loc[:10000]


# Separate target feature 

# In[176]:


y = train_df['target']
train_df.drop('target', axis=1, inplace=True)
times = ['time%s' % i for i in range(1, 11)]
train_df[times] = train_df[times].apply(pd.to_datetime)
test_df[times] = test_df[times].apply(pd.to_datetime)

train_df.head()

# In[177]:


# загрузим словарик сайтов
with open(r"/home/andrei/Desktop/alice_kaggle/site_dic.pkl", "rb") as input_file:
    site_dict = pickle.load(input_file)

# датафрейм словарика сайтов
sites_dict_df = pd.DataFrame(list(site_dict.keys()),
                             index=list(site_dict.values()),
                             columns=['site'])
print(u'всего сайтов:', sites_dict_df.shape[0])
sites_dict_df.head()

# In[178]:


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

# In[179]:


train_tfidf = [word_vec.transform(train_df['text_col'].values.astype('U'))]
train_X = hstack(train_tfidf, format='csr')
print(train_X.shape)
test_tfidf = [word_vec.transform(test_df['text_col'].values.astype('U'))]
test_X = hstack(test_tfidf, format='csr')
print(test_X.shape)

# Build Tf-Idf features based on sites. You can use `ngram_range`=(1, 3) and `max_features`=100000 or more

# Add features based on the session start time: hour, whether it's morning, day or night and so on.

# Scale this features and combine then with Tf-Idf based on sites (you'll need `scipy.sparse.hstack`)

# In[180]:


alice_common = dict()
for i in range(1, 11):
    a = train_df[y == 0]['site_column%s' % i].value_counts()
    for k, v in dict(a).items():
        alice_common.setdefault(k, 0)
        alice_common[k] += v
alice_common = list(alice_common.items())
alice_common.sort(key=lambda x: x[1], reverse=True)
alice_common = set([_[0] for _ in alice_common[:30]])


def get_part_from_hour(h):
    if h < 6:
        return 0
    if h < 12:
        return 1
    if h < 18:
        return 2
    return 3


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

    for i in range(2, 7):
        df['delta%s' % (i - 1)] = (df['time%s' % i] - df['time%s' % (i - 1)]).dt.total_seconds()

    df['duration'] = df.apply(get_session_duration, axis=1)

    df['count_good'] = df.apply(get_count_of_good_sites, axis=1)

    return df


def get_scaled_features(df):
    scalable_columns = ['hour', 'delta1', 'delta2', 'delta3',
                        'delta4', 'delta5', 'duration', 'count_good']
    return StandardScaler().fit_transform(df[scalable_columns].fillna(-1))


def get_part_of_day(df):
    df['part_of_day'] = np.array(list(map(lambda v: get_part_from_hour(v), df['time1'].dt.hour)))
    return OneHotEncoder().fit_transform(df[['part_of_day']])


def get_extra_features(df):
    return get_scaled_features(add_extra_features(df))


# Perform cross-validation with logistic regression.

# In[181]:


# get_part_of_day(train_df)
train_X = hstack([train_X, get_extra_features(train_df), get_part_of_day(train_df)], format='csr')
test_X = hstack([test_X, get_extra_features(test_df), get_part_of_day(test_df)], format='csr')

# Make prediction for the test set and form a submission file.

# In[182]:


logit = LogisticRegression(n_jobs=-1, random_state=17)

log_params = {'C': list(np.power(10.0, np.arange(-4, 4)))}
skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=17)
lgt_grid = GridSearchCV(logit, log_params, cv=skf, scoring='roc_auc', verbose=2)
lgt_grid.fit(train_X, y)

# In[183]:


lgt_grid.best_score_

# In[184]:


test_pred = lgt_grid.predict_proba(test_X)[:, 1]


# In[185]:


def write_to_submission_file(predicted_labels, out_file,
                             target='target', index_label="session_id"):
    predicted_df = pd.DataFrame(predicted_labels,
                                index=np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)


# In[186]:


write_to_submission_file(test_pred, "assignment6_alice_submission.csv")
