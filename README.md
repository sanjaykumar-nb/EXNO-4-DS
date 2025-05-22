# EXNO:4-Feature Scaling and Selection
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![image](https://github.com/user-attachments/assets/2a2dca77-37f9-4888-8f6b-786091dfaffd)
```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/cab5da29-dfee-45df-bbf7-2786b1385825)
```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/user-attachments/assets/8d56f93f-b58f-4c0a-b322-f6462223152b)
```
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/user-attachments/assets/82800cce-e6fc-42fa-b508-263de8895b37)
```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/user-attachments/assets/ff443f16-4b1b-4d47-b227-9fe622c4b5c5)
```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/user-attachments/assets/821a002c-0a2b-4b08-a354-ecb52b843e5e)
```
data2
```
![image](https://github.com/user-attachments/assets/ec758f58-c3d3-42b0-b7a5-86ec18cc650d)
```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/user-attachments/assets/42df36c8-6e45-4d59-8cbc-c3bcef07deac)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/user-attachments/assets/e4b19d16-1ffc-4168-a464-109152f11ba2)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/user-attachments/assets/ce47d4d3-913a-4a70-8480-92fd608b846f)
```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/user-attachments/assets/67412d9a-b567-4080-847b-6aa22bf58137)
```
x=new_data[features].values
print(x)
```
![image](https://github.com/user-attachments/assets/00aac4ea-3f6a-4f29-b2bb-240dcc013278)
```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/user-attachments/assets/50bcee3c-56a0-4a2f-bdeb-b1bcd5ee993e)
```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/user-attachments/assets/07ce3deb-c179-41d7-9cc7-c3172dcce92b)
```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/user-attachments/assets/7d3d8551-0aba-4d2b-bf25-3594074fa72a)
```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![image](https://github.com/user-attachments/assets/c5e84c67-092c-44ef-bfc5-1f89ad86fd60)
```
data.shape
```
![image](https://github.com/user-attachments/assets/f00e55b7-c1c5-486b-8134-7dc5dcab6bde)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/dcd86342-d16f-4f9f-94f8-51c1481e435e)
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/user-attachments/assets/1731ae20-26ff-498a-b480-29abc540872d)
```
tips.time.unique()
```
![image](https://github.com/user-attachments/assets/2f39f17d-d37c-47db-8626-d3b61c3bf79f)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/user-attachments/assets/e7b1abeb-c7ad-4d1a-a9ab-a0a9181815b6)
```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![image](https://github.com/user-attachments/assets/5b61fb88-63e3-484d-beee-8d811e01097f)


# RESULT:
       Thus, Feature selection and Feature scaling has been used on thegiven dataset.
