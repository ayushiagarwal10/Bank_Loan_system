
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
data = pd.read_csv("loan.csv")
import seaborn as sns
from sklearn import preprocessing

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rc("font", size=10)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


# In[2]:


data.info()


# In[3]:


numerical1 = data.dtypes[data.dtypes == int].index

print(numerical1)


# In[4]:


numerical = data.dtypes[data.dtypes == float].index

print(numerical)


# In[4]:


data.head()


# In[5]:


data.shape


# In[6]:


categorical = data.dtypes[data.dtypes == "object"].index

print(categorical)


# In[7]:


data[categorical].describe


# In[8]:


data.apply(lambda x: sum(x.isnull()))


# In[9]:


data = data.drop(data.columns[data.apply(lambda col: col.isnull().sum()/len(data) > 0.5)], axis=1)


# In[10]:


data.shape


# In[11]:


data.apply(lambda x: sum(x.isnull()))


# In[12]:


data = data.dropna(axis=0,subset =['id'])
data = data.dropna(axis=0,subset =['member_id'])
data = data.dropna(axis=0,subset =['desc'])
data = data.dropna(axis=0,subset =['emp_title'])



# In[13]:


data.shape


# In[14]:


data.apply(lambda x: sum(x.isnull()))


# In[15]:


data.head()


# In[16]:


data.shape


# In[17]:


data.info()


# In[18]:


data["funded_amnt"].plot(kind="box", figsize=(9,9))
data["installment"].plot(kind="hist", figsize=(9,9))


# In[19]:


data["installment"].plot(kind="hist", figsize=(9,9))


# In[20]:


data['pub_rec'].value_counts()


# In[21]:


data['emp_length'].value_counts()


# In[22]:


data['chargeoff_within_12_mths'].value_counts()


# In[23]:


data=data.drop(['chargeoff_within_12_mths'] , axis=1)


# In[24]:


data.shape


# In[25]:


data['collections_12_mths_ex_med'].value_counts()


# In[26]:


data=data.drop(['collections_12_mths_ex_med'] , axis=1)


# In[27]:


data['loan_status'].value_counts()


# In[28]:


data["pub_rec_bankruptcies"].value_counts()


# In[29]:


data["pub_rec_bankruptcies"] =data["pub_rec_bankruptcies"].fillna(data["pub_rec_bankruptcies"].median())


# In[30]:


numerical = data.dtypes[data.dtypes == float]

print(numerical)


# In[31]:


data["tax_liens"].value_counts()


# In[32]:


data["tax_liens"] =data["tax_liens"].fillna(data["tax_liens"].median())


# In[33]:


data = data.drop(data.columns[data.apply(lambda y :y.value_counts==0)],axis=1)


# In[34]:


data.shape


# In[35]:


data["delinq_amnt"].value_counts()


# In[36]:


data["delinq_amnt"] =data["delinq_amnt"].fillna(data["delinq_amnt"].median())


# In[37]:


data["acc_now_delinq"].value_counts()


# In[38]:


data["acc_now_delinq"] =data["acc_now_delinq"].fillna(data["acc_now_delinq"].median())


# In[39]:


data.apply(lambda x: sum(x.isnull()))


# In[40]:


data["acc_now_delinq"] =data["acc_now_delinq"].fillna(data["acc_now_delinq"].median())


# In[41]:


data["open_acc"] =data["open_acc"].fillna(data["open_acc"].mean())
data["total_acc"] =data["total_acc"].fillna(data["total_acc"].mean())
data["delinq_2yrs"] =data["delinq_2yrs"].fillna(data["delinq_2yrs"].median())


# In[42]:


data["open_acc"].value_counts()
data["delinq_2yrs"].value_counts()


# In[43]:


data["emp_length"].value_counts()


# In[44]:


data["emp_length"] =data["emp_length"].fillna("10+ years")


# In[45]:


print(categorical)


# In[46]:


data.apply(lambda y :sum(y.isnull()))


# In[47]:


data["purpose"].value_counts()


# In[48]:


data["purpose"] =data["purpose"].fillna("debt_consolidation")


# In[49]:


data["title"].value_counts()


# In[50]:


data["title"] =data["title"].fillna("Debt Consolidation")


# In[51]:


data["revol_util"].value_counts()


# In[52]:


data["revol_util"] =data["revol_util"].fillna("0%")


# In[53]:


data.shape


# In[54]:


data.apply(lambda a:sum(a.isnull()))


# In[55]:


data["pub_rec"].value_counts()


# In[56]:


data["pub_rec"]= data["pub_rec"].fillna(data["pub_rec"].mean())


# In[57]:


data =data.drop(['last_pymnt_d','last_credit_pull_d','inq_last_6mths','earliest_cr_line'],axis=1)


# In[58]:


print(categorical)


# In[59]:


corr = data.corr()
plt.figure(figsize = (16,12))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .82})


# In[60]:


data2 = pd.get_dummies(data, columns =['term', 'int_rate', 'grade', 'sub_grade', 
       'emp_length', 'home_ownership', 'verification_status', 
       'loan_status', 'pymnt_plan'])


# In[61]:


print(data2)


# In[62]:


data2.shape


# In[63]:


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[64]:


X_features = data[['member_id', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv',
       'installment', 'annual_inc', 'dti', 'delinq_2yrs',
        'open_acc', 'pub_rec', 'revol_bal', 'total_acc',
       'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',
       'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries',
       'collection_recovery_fee', 'last_pymnt_amnt', 
        'policy_code', 'acc_now_delinq', 'delinq_amnt',
       'pub_rec_bankruptcies', 'tax_liens']].values


# In[65]:


Y_target = data['loan_status'].values


# In[66]:


X_train , X_test , Y_train ,Y_test =model_selection.train_test_split(X_features,Y_target,test_size = 0.2,random_state=42)


# In[67]:


linearmodel=LogisticRegression()
linearmodel.fit(X_train,Y_train)

my_prediction = logistic_model.predict(X_test)
# In[68]:


my_prediction = linearmodel.predict(X_test)


# In[69]:


my_prediction = linearmodel.predict(X_test)


# In[70]:


print(my_prediction)


# In[71]:


linearmodel.classes_


# In[72]:


sns.countplot(x="loan_status",data=data)
plt.show()


# In[73]:


sns.countplot(x="home_ownership",data=data)


# In[74]:


sns.countplot(x="grade",data= data)


# In[75]:


sns.countplot(x="term",data= data)


# In[76]:


sns.countplot(x="pymnt_plan",data=data)


# In[77]:


sns.countplot(x="verification_status",data=data)


# In[78]:


data["verification_status"][data["verification_status"] == "Source Verified"] = 0
data["verification_status"][data["verification_status"] == "Not Verified"] = 1
data["verification_status"][data["verification_status"] == "Verified"] = 2


data["term"][data["term"] == "36months"] = 0
data["term"][data["term"] == "60months"] = 1

data["grade"][data["grade"] == "A"] = 0
data["grade"][data["grade"] == "B"] = 1
data["grade"][data["grade"] == "C"] = 2
data["grade"][data["grade"] == "D"] = 3

data["grade"][data["grade"] == "E"] = 4
data["grade"][data["grade"] == "F"] = 5

data["grade"][data["grade"] == "G"] = 6



data["home_ownership"][data["home_ownership"] == "RENT"] = 0
data["home_ownership"][data["home_ownership"] == "MORTGAGE"] = 1
data["home_ownership"][data["home_ownership"] == "OWN"] = 2
data["home_ownership"][data["home_ownership"] == "OTHER"] = 3
data["home_ownership"][data["home_ownership"] == "NONE"] = 4

data["pymnt_plan"][data["pymnt_plan"] == "n"] = 0
data["pymnt_plan"][data["pymnt_plan"] == "y"] = 1


# In[79]:


model4=KNeighborsClassifier(n_neighbors=5)
X_features_new=data[[ 'grade', 'home_ownership', 'verification_status','pymnt_plan','loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'installment', 'annual_inc', 'dti', 'delinq_2yrs',  'open_acc', 'pub_rec', 'revol_bal', 'total_acc', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_amnt',  'policy_code', 'acc_now_delinq', 'delinq_amnt', 'pub_rec_bankruptcies', 'tax_liens']].values
Y_new_target=data['loan_status'].values
X_train , X_test , Y_train ,Y_test =model_selection.train_test_split(X_features_new,Y_new_target,test_size = 0.2,random_state=42)


# In[80]:


model=model4.fit(X_train,Y_train)
Y_predictions=model4.predict(X_test)


# In[81]:


data['home_ownership'].value_counts()


# In[82]:


print(Y_predictions)
from sklearn import metrics


# In[83]:


m_acc = metrics.accuracy_score(Y_test, Y_predictions)

print("Accuracy:",m_acc)


# In[84]:


data['pymnt_plan'].value_counts()


# In[85]:


X_features_new=data[[ 'grade', 'home_ownership', 'verification_status','pymnt_plan','loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'installment', 'annual_inc', 'dti', 'delinq_2yrs', 'open_acc', 'pub_rec', 'revol_bal', 'total_acc', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_amnt', 'policy_code', 'acc_now_delinq', 'delinq_amnt', 'pub_rec_bankruptcies', 'tax_liens']].values
Y_new_target=data['loan_status'].values
model5=DecisionTreeClassifier(max_depth=13)
X_train , X_test , Y_train ,Y_test =model_selection.train_test_split(X_features_new,Y_new_target,test_size = 0.2,random_state=42)


# In[86]:


model5.fit(X_train,Y_train)


# In[87]:


Y_predictions1=model5.predict(X_test)


# In[88]:


m_acc = metrics.accuracy_score(Y_test, Y_predictions1)

print("Accuracy:",m_acc)


# In[89]:


from sklearn.ensemble import RandomForestClassifier
X_features_new=data[[ 'grade', 'home_ownership', 'verification_status','pymnt_plan','loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'installment', 'annual_inc', 'dti', 'delinq_2yrs',  'open_acc', 'pub_rec', 'revol_bal', 'total_acc', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_amnt', 'policy_code', 'acc_now_delinq', 'delinq_amnt', 'pub_rec_bankruptcies', 'tax_liens']].values
Y_new_target=data['loan_status'].values
model5=DecisionTreeClassifier(max_depth=13)
X_train , X_test , Y_train ,Y_test =model_selection.train_test_split(X_features_new,Y_new_target,test_size = 0.2,random_state=42)


# In[90]:


forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
forest.fit(X_train,Y_train)


# In[91]:


Y_predictions2 = forest.predict(X_test)


# In[92]:


m_acc = metrics.accuracy_score(Y_test,Y_predictions2)


# In[93]:


print("Accuracy:",m_acc)


# In[94]:


print(metrics.classification_report(y_true=Y_test,
                              y_pred=Y_predictions2) )


# In[95]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_predictions2,Y_test)


# In[96]:


print(confusion_matrix)


# In[97]:


from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
print(cross_val_score(forest, X_features_new, Y_new_target,cv=10))


# In[98]:


from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


# In[99]:


def plot_learning_curve(estimator, title, X_features_new, Y_new_target, ylim=None, cv=None,n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
  
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X_features_new, Y_new_target, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")


# In[105]:


title = "Learning Curves"
cv = ShuffleSplit(n_splits=1, test_size=0.25, random_state=12)
estimator = forest
plot_learning_curve(estimator, title, X_features_new, Y_new_target, (1,1), cv=cv, n_jobs=10)

plt.show()


# In[235]:


data.corr()


# In[236]:


data.head()


# In[237]:


from sklearn.naive_bayes import GaussianNB


# In[239]:


model6 =GaussianNB()
X_features_new=data[[ 'grade', 'home_ownership', 'verification_status','pymnt_plan','loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'installment', 'annual_inc', 'dti', 'delinq_2yrs', 'open_acc', 'pub_rec', 'revol_bal', 'total_acc', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_amnt',  'policy_code', 'acc_now_delinq', 'delinq_amnt', 'pub_rec_bankruptcies', 'tax_liens']].values
Y_new_target=data['loan_status'].values
X_train , X_test , Y_train ,Y_test =model_selection.train_test_split(X_features_new,Y_new_target,test_size = 0.2,random_state=42)


# In[240]:


model6.fit(X_train,Y_train)


# In[241]:


Y_predictions3 = model6.predict(X_test)


# In[242]:


print(Y_predictions3)


# In[243]:


m_acc = metrics.accuracy_score(Y_test,Y_predictions3)


# In[244]:


print(m_acc)


# In[246]:


y_prob=forest.predict_proba(X=X_test)


# In[247]:


print(y_prob)

