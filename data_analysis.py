"""
Read the dataset and analyse the data.
"""

import pandas as pd
import numpy as np


### READ DATASET ###
df = pd.read_csv('DatasetML.csv')

### ANALYSE DATASET ###
print (df.dtypes) # only 'int64' and 'object'

### CHECK IF THERE IS NULL VALUES ###
print (df.isnull().sum())  # no null values, no imputation needed

### ANALYSE EACH FEATURE ###
row1 = []
row2 = []
for name in df.columns:
#    print(name)
    if name=='LABEL':
        continue
    if df[name].dtype=='object':
        unique_categories = set(df[name])
        count_categories = len(unique_categories)
        row1.append(unique_categories)
        row2.append(count_categories)
#        print('Categories:', unique_categories)
#        print('Total of Categories:', count_categories, '\n')
    else:
        col_mean = np.mean(df[name])
        col_std = np.std(df[name])
        row1.append(col_mean)
        row2.append(col_std)
#        print('Mean:', col_mean)
#        print('Std:', col_std, '\n')

df_analysis = pd.DataFrame(np.array([row1, row2]), columns=df.columns[0:-1])
  
### FIND OUTLIERS ###
outliers = []
for name in df.columns:
    if name=='LABEL':
        continue
    if df[name].dtype=='object':
        outliers.append('object_type')
        continue
    col = abs(df[name]-df_analysis[name][0]) - 4*df_analysis[name][1]
    count_positives = sum(col>0)
    outliers.append(count_positives)

### CHECK BALANCE ###
label = df['LABEL']
str_label = list(map(str,label))
print('Classes:', set(str_label)) # Classes: {'2', '1'}
label = label-1
n_class2 = sum(label)
n_class1 = len(label)-n_class2
print('Class 1:', n_class1) # Class 1: 700
print('Class 2:', n_class2) # Class 2: 300
