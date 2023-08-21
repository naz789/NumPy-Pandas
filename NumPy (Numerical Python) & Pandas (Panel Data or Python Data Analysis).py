#!/usr/bin/env python
# coding: utf-8

# # NumPy: Package for multidimensional array (Machine learning library)
# 
# NumPy is the shortform for numerical python.
# NumPy aims to provide an array that is upto 50x faster than traditional python lists.
# np is the keyword for it. 

# In[1]:


import numpy as np


# In[3]:


simple_list=[6,7,8]
np.array(simple_list)


# In[5]:


arr=np.array([1,2,3])
arr


# In[7]:


list_of_lists=[[1,2,3],[4,5,6],[7,8,9]]
np.array(list_of_lists)


# In[6]:


np.arange(5,10)


# In[11]:


np.arange(1,100)


# In[12]:


np.arange(1,67,5)


# In[13]:


np.arange(5)


# In[14]:


np.zeros(10)


# In[15]:


np.zeros(10,int)


# In[16]:


np.ones((2,6))


# In[17]:


np.ones(100)


# In[18]:


np.ones(10,int)


# In[19]:


np.zeros((2,5),int)


# In[20]:


np.linspace(0,2,5)


# In[22]:


np.linspace(0,8,20)


# In[23]:


np.eye((7))


# In[25]:


arr=np.random.rand(2,4)
arr


# In[27]:


np.random.randint(2,100)


# In[30]:


np.random.randint(20,34,14)


# In[44]:


sample_arr=np.arange(20)
sample_arr


# In[34]:


rand_arr=np.random.randint(0,30,15)
rand_arr


# In[40]:


sample_array=(6,9,12)
sample_array


# In[46]:


sample_arr.reshape(5,4)


# In[53]:


#Cannot be reshaped in rows and columns of 20
sample_array.reshape(4,4)


# In[55]:


rand_arr.max()


# In[57]:


#It will show index of the maximum value
rand_arr.argmax()


# In[59]:


#Gives identity matrix
a=np.eye(9)
a


# In[60]:


#Will give identity transpose(interchange of position from rows to column and vice versa)
a.T


# In[62]:


a=np.random.rand(4,5)
a


# In[63]:


#Will give identity transpose(interchange of position from rows to column and vice versa)
a.T


# In[66]:


sample_array[1:3]


# In[5]:


sample_array=np.arange(10,21)
sample_array


# In[69]:


sample_array[0]


# In[70]:


sample_array[2:5]


# In[71]:


sample_array[1:4]=100
sample_array


# In[10]:


sample_array=np.arange(10,21)
sample_array


# In[76]:


sample_array[0:7]


# In[15]:


subset_sample_array=sample_array[0:6]
subset_sample_array


# In[78]:


subset_sample_array[:]=1001
subset_sample_array


# # TWO DIMENSIONAL ARRAY
# 
# A 2D Array is an array of arrays that can be represented in the form of matrix form like rows & columns.

# In[2]:


import numpy as np


# In[79]:


sample_matrix=np.array([[30,40,50,60],[21,31,41,51],[93,83,73,63]])
sample_matrix


# In[80]:


sample_matrix[1,2]


# In[81]:


sample_matrix[2,:]


# In[82]:


sample_matrix[2]


# In[83]:


sample_matrix[:,(3,2)]


# # SELECTION TECHNIQUES

# In[19]:


sample_arr= np.arange(1,20)
sample_arr


# In[21]:


sample_arr


# In[22]:


sample_arr+sample_arr


# In[23]:


np.exp(sample_arr) #exponential


# In[24]:


np.sqrt(sample_arr) #square root


# In[25]:


np.log(sample_arr) #logarithm


# In[26]:


np.max(sample_arr) #maximum value


# In[27]:


np.min(sample_arr) #minimum value


# In[28]:


np.argmax(sample_arr) #index of maximum value


# In[29]:


np.argmin(sample_arr) #index of maximum value


# In[30]:


np.square(sample_arr) #square of the value


# In[ ]:


np.std(sample_arr) #standardization


# In[31]:


np.var(sample_arr) #variant


# In[32]:


np.mean(sample_arr)


# In[34]:


array=np.random.randn(3,4)
array


# In[35]:


np.round(array,decimals=3)


# In[38]:


sports=np.array(["Golf", "Cricket", "Badminton", "Table tennis", "Cricket"])
np.unique(sports)


# # PANDAS (Panel data or Python Data Analysis)
# 
# Its a python library used for working with Data sets. 
# It has functions for analyzing, cleaning, exploring and manipulating data.
# pd is the keyword for it.
# 

# In[14]:


import pandas as pd
import numpy as np


# # PANDAS dataframe and indexing

# In[21]:


#series we use for single column
sports1=pd.Series([1,2,3,4], index=['Cricket','badminton','table tennis','golf'])
sports1


# In[17]:


sports1['badminton']


# In[41]:


sports2=pd.Series([11,2,3,4,6], index=['Ludo','badminton','table tennis','golf','basketball'])
sports2


# In[19]:


sports1+sports2


# In[20]:


import pandas as pd
import numpy as np


# In[23]:


#DataFrame is used for multiple rows and column
df1=pd.DataFrame(np.random.rand(8,5), index='A B C D E F G H'.split(),columns='Score1 Score2 Score3 Score4 Score5'.split())
df1


# In[25]:


df1["Score1"]


# In[28]:


df1[["Score1", "Score2", "Score3"]]


# In[29]:


#we can add two or more columns to get another column as well.

df1['Score6']= df1['Score1']+df1['Score2']+df1['Score3']+df1['Score4']+df1['Score5']
df1


# In[30]:


df2= {'ID':['101','102','103','107','176'],'Name':['John','Mercy','Akash','Kevin','Sally'],'Profit':[20,30,40,50,10]}
df=pd.DataFrame(df2)
df


# In[31]:


df["ID"]


# In[32]:


df[["ID","Name","Profit"]]


# In[34]:


#axis means column
df=df.drop("ID",axis=1)
df


# In[36]:


df.drop(3)


# # END
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
