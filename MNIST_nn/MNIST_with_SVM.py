
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


labeled_images = pd.read_csv('/home/andrei/PycharmProjects/ds-ml/MNIST_nn/train.csv')
images = labeled_images.iloc[0:5000,1:]
labels = labeled_images.iloc[0:5000,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)


# In[29]:


image_id=3451
img=train_images.iloc[image_id].as_matrix()
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[image_id,])


# In[31]:


plt.hist(train_images.iloc[image_id])


# In[32]:


clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)


# In[35]:


# binary representation
threshold = 10
test_images[test_images>threshold]=1
train_images[train_images>threshold]=1

img=train_images.iloc[image_id].as_matrix().reshape((28,28))
plt.imshow(img,cmap='binary')
plt.title(train_labels.iloc[image_id])


# In[36]:


plt.hist(train_images.iloc[image_id])


# In[44]:


clf = svm.SVC(C=7, gamma=0.009)
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)


# In[41]:


test_data=pd.read_csv('/home/andrei/PycharmProjects/ds-ml/MNIST_nn/test.csv')
test_data[test_data>10]=1
results=clf.predict(test_data)

