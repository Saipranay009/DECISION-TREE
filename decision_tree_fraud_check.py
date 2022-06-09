# -*- coding: utf-8 -*-
"""
Created on Thu May 12 17:41:05 2022

@author: Sai pranay
"""

#-----------------------IMPORTING_THE_DATA_SET---------------------------------


import pandas as pd
dt_fc = pd.read_csv("E:\\DATA_SCIENCE_ASS\\DECISION TREE\\Fraud_check.csv")
print(dt_fc)
dt_fc.info()
list(dt_fc)
dt_fc.describe()

import seaborn as sns

sns.distplot(dt_fc['City.Population'])
sns.distplot(dt_fc['Work.Experience'])

sns.countplot(dt_fc['Undergrad'])
sns.countplot(dt_fc['Marital.Status'])
sns.countplot(dt_fc['Urban'])
sns.countplot(dt_fc['Category'])


#-----------------working_on_label_encoding------------------------------------

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
dt_fc['Undergrad_1']=LE.fit_transform(dt_fc['Undergrad'])
dt_fc['Marital.Status_1']=LE.fit_transform(dt_fc['Marital.Status'])
dt_fc['Urban_1']=LE.fit_transform(dt_fc['Urban'])

list(dt_fc)

#-------------------------Dropping---------------------------------------------

dt_fc_new = dt_fc.drop(['Undergrad','Marital.Status','Urban'],axis=1)
list(dt_fc_new)


ts1 = dt_fc_new.iloc[:,0:3]
ts1

ts2 = dt_fc_new.iloc[:,3:6]
list(ts2)

ts3= dt_fc_new.iloc[:,1:3]
list(ts3)

#from sklearn.preprocessing import StandardScaler
#st=StandardScaler()
#x1=st.fit_transform(ts3)
#x1
#x11=pd.DataFrame(x1)
#x11


dt_fc_r = pd.concat([ts3,ts2],axis = 1)
dt_fc_r.shape

#-----------------------------------splitting----------------------------------

x = dt_fc_r.iloc[:,:]
x

y = ts1.iloc[:,0]
y


Y1=[]
for i in range(0,600,1):
    if y.iloc[i,]<=30000:
        print('risky')
        Y1.append('risky')
    else:
        print('good')
        Y1.append('good')

y2 = pd.DataFrame(Y1)
y2

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y2, test_size=0.25,stratify=y2,random_state=99)

X_train.shape
X_test.shape
Y_train.shape
Y_test.shape



# ---------------------fit the model _via (criterion='gini')-------------------
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion='gini')
DT.fit(X_train,Y_train)

Y_pred = DT.predict(X_test)
Y_pred

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y_test,Y_pred)
cm
ac = accuracy_score(Y_test,Y_pred)
ac


#-------------------------------------------------------------------------------------


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y2, test_size=0.25,stratify=y2,random_state=0)

X_train.shape
X_test.shape
Y_train.shape
Y_test.shape



# ---------------------fit the model _via (criterion='entropy')----------------
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion='entropy',max_depth=(7))
DT.fit(X_train,Y_train)

Y_pred = DT.predict(X_test)
Y_pred

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y_test,Y_pred)
cm
ac = accuracy_score(Y_test,Y_pred)
ac


#Tree
import matplotlib.pyplot as plt
from sklearn import tree
tr=tree.plot_tree(DT,filled=True,fontsize=6)


DT.tree_.node_count
DT.tree_.max_depth
