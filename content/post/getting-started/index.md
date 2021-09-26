---
title: Titanic Predictions
subtitle: The Titanic data set is a well known data set used in Kaggle
  challenges.  This blog post will discuss the tutorial provided by Kaggle for
  the Titanic data set, as well as some additional contributions using Logistic
  Regression.
date: 2020-12-13T00:00:00.000Z
summary: Welcome 👋 We know that first impressions are important, so we've
  populated your new site with some initial content to help you get familiar
  with everything in no time.
draft: false
featured: false
authors:
  - admin
lastmod: 2020-12-13T00:00:00.000Z
tags: []
categories: []
projects: []
image:
  caption: ""
  focal_point: ""
  placement: 2
  preview_only: false
---
```python
# All of this code until the section marked "My Contribution" was taken from 
# https://www.kaggle.com/alexisbcook/titanic-tutorial for educational purposes.

import numpy as np 
import pandas as pd 
```


```python
# Loading the training data set
train_data = pd.read_csv(r"C:\Users\zachc\OneDrive\Desktop\train.csv")
train_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Loading the testing data set

test_data = pd.read_csv(r"C:\Users\zachc\OneDrive\Desktop\test.csv")
test_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>Myles, Mr. Thomas Francis</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>9.6875</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>Wirz, Mr. Albert</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>315154</td>
      <td>8.6625</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>3101298</td>
      <td>12.2875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Looking at the percentage of passengers who were women that survived

women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)
```

    % of women who survived: 0.7420382165605095
    


```python
# Looking at the percentage of passengers who were men that survived

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)
```

    % of men who survived: 0.18890814558058924
    


```python
# Using a random forest to create a classifier for survival on the Titanic

from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])
print(X)

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('Tutorial_Submission.csv', index=False)
print("Your submission was successfully saved!")
```

         Pclass  SibSp  Parch  Sex_female  Sex_male
    0         3      1      0           0         1
    1         1      1      0           1         0
    2         3      0      0           1         0
    3         1      1      0           1         0
    4         3      0      0           0         1
    ..      ...    ...    ...         ...       ...
    886       2      0      0           0         1
    887       1      0      0           1         0
    888       3      1      2           1         0
    889       1      0      0           0         1
    890       3      0      0           0         1
    
    [891 rows x 5 columns]
    Your submission was successfully saved!
    

# My Contribution
In this section, we'll conduct a Logistic Regression to create a model which will act as our classifier.

## Before we begin, we'll parse through our training data set to see if there exists any missing values and what variables we'll use for training and testing.  


```python
# Importing some necessary packages
import matplotlib.pyplot as plt

# Checking to see what features in the training data set contain missing values.
train_data.isnull().sum()
test_data.isnull().sum()

# We'll check the mean and median of Fare
print(test_data["Fare"].describe())

# Checking the distribution of Fare.  
fare_ar = test_data["Fare"]
plt.hist(fare_ar,bins = 15)

# # The distribution of Fare is skewed the right so we'll use the median value to replace the 
# missing Fare value for our testing data.
fare_test = test_data["Fare"]; fare_med = fare_ar.median(); 
fare_test = fare_test.fillna(fare_med)
print(fare_test)
```

    count    417.000000
    mean      35.627188
    std       55.907576
    min        0.000000
    25%        7.895800
    50%       14.454200
    75%       31.500000
    max      512.329200
    Name: Fare, dtype: float64
    0        7.8292
    1        7.0000
    2        9.6875
    3        8.6625
    4       12.2875
             ...   
    413      8.0500
    414    108.9000
    415      7.2500
    416      8.0500
    417     22.3583
    Name: Fare, Length: 418, dtype: float64
    

    C:\Users\zachc\anaconda3\lib\site-packages\numpy\lib\histograms.py:839: RuntimeWarning: invalid value encountered in greater_equal
      keep = (tmp_a >= first_edge)
    C:\Users\zachc\anaconda3\lib\site-packages\numpy\lib\histograms.py:840: RuntimeWarning: invalid value encountered in less_equal
      keep &= (tmp_a <= last_edge)
    


![png](output_8_2.png)



```python
from sklearn.linear_model import LogisticRegression 

print(train_data)
fare_train = pd.DataFrame(train_data["Fare"])
fare_test = pd.DataFrame(fare_test)
print(fare_train)

# Adding the variable "Fare" to the set of features for training and testing.
# X_train_new = pd.concat([X,fare_train],axis = 1)
# X_test_new = pd.concat([X_test,fare_test],axis = 1)

X_train_new = X; X_test_new = X_test; 

# Creating our model based on training data with and without penalty.
# The penalty term is using l1 and l2 regularization.  
# The l1 regularizer will help promote sparsity in the coefficients for our predictor variables while the 
# l2 reguluarizer helps with any potential outliers in our predictor variables.  We'll compare the predictions between 
# the three outputs.  
lr = LogisticRegression()
lrl1 = LogisticRegression(penalty = 'l1', solver = 'liblinear') 
lrl2 = LogisticRegression(penalty = 'l2') 
trained_model_np = lr.fit(X_train_new,y)
trained_model_l1 = lrl1.fit(X_train_new,y)
trained_model_l2 = lrl2.fit(X_train_new,y)

# Making predictions on testing data.
test_predict_np = lr.predict(X_test_new)
test_predict_l1 = lrl1.predict(X_test_new)
test_predict_l2 = lrl2.predict(X_test_new)

# Exporting necessary columns to a CSV file to submit to Kaggle.
my_contr_np = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': test_predict_np})
my_contr_l1 = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': test_predict_l1})
my_contr_l2 = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': test_predict_l2})
my_contr_np.to_csv('My_Contribution_NP_2.csv', index= False )
#my_contr_np.to_csv('My_Contribution_l1.csv', index= False )
#my_contr_np.to_csv('My_Contribution_l2.csv', index= False )

print(test_predict_np)
#print(test_predict_l1)
#print(test_predict_l2)
```

         PassengerId  Survived  Pclass  \
    0              1         0       3   
    1              2         1       1   
    2              3         1       3   
    3              4         1       1   
    4              5         0       3   
    ..           ...       ...     ...   
    886          887         0       2   
    887          888         1       1   
    888          889         0       3   
    889          890         1       1   
    890          891         0       3   
    
                                                      Name     Sex   Age  SibSp  \
    0                              Braund, Mr. Owen Harris    male  22.0      1   
    1    Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
    2                               Heikkinen, Miss. Laina  female  26.0      0   
    3         Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
    4                             Allen, Mr. William Henry    male  35.0      0   
    ..                                                 ...     ...   ...    ...   
    886                              Montvila, Rev. Juozas    male  27.0      0   
    887                       Graham, Miss. Margaret Edith  female  19.0      0   
    888           Johnston, Miss. Catherine Helen "Carrie"  female   NaN      1   
    889                              Behr, Mr. Karl Howell    male  26.0      0   
    890                                Dooley, Mr. Patrick    male  32.0      0   
    
         Parch            Ticket     Fare Cabin Embarked  
    0        0         A/5 21171   7.2500   NaN        S  
    1        0          PC 17599  71.2833   C85        C  
    2        0  STON/O2. 3101282   7.9250   NaN        S  
    3        0            113803  53.1000  C123        S  
    4        0            373450   8.0500   NaN        S  
    ..     ...               ...      ...   ...      ...  
    886      0            211536  13.0000   NaN        S  
    887      0            112053  30.0000   B42        S  
    888      2        W./C. 6607  23.4500   NaN        S  
    889      0            111369  30.0000  C148        C  
    890      0            370376   7.7500   NaN        Q  
    
    [891 rows x 12 columns]
            Fare
    0     7.2500
    1    71.2833
    2     7.9250
    3    53.1000
    4     8.0500
    ..       ...
    886  13.0000
    887  30.0000
    888  23.4500
    889  30.0000
    890   7.7500
    
    [891 rows x 1 columns]
    [0 1 0 0 1 0 1 0 1 0 0 0 1 0 1 1 0 0 1 1 0 0 1 0 1 0 1 0 0 0 0 0 1 1 0 0 1
     1 0 0 0 0 0 1 1 0 0 0 1 1 0 0 1 1 0 0 0 0 0 1 0 0 0 1 0 1 1 0 0 1 1 0 1 0
     1 0 0 1 0 1 0 0 0 0 0 0 1 1 1 0 1 0 1 0 0 0 1 0 1 0 1 0 0 0 1 0 0 0 0 0 0
     1 1 1 1 0 0 1 0 1 1 0 1 0 0 1 0 1 0 0 0 0 1 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0
     0 0 1 0 0 1 0 0 1 1 0 1 1 0 1 0 0 1 0 0 1 1 0 0 0 0 0 1 1 0 1 1 0 0 1 0 1
     0 1 0 0 0 0 0 0 0 0 0 0 1 0 1 1 0 0 1 0 0 1 0 1 0 0 0 0 1 0 0 1 0 1 0 1 0
     1 0 1 1 0 1 0 0 0 1 0 0 0 0 0 0 1 1 1 1 0 0 0 0 1 0 1 1 1 0 0 0 0 0 0 0 1
     0 0 0 1 1 0 0 0 0 1 0 0 0 1 1 0 1 0 0 0 0 1 0 1 1 1 0 0 0 0 0 0 1 0 0 0 0
     1 0 0 0 0 0 0 0 1 1 0 0 0 1 0 0 0 1 1 1 0 0 0 0 0 0 0 0 1 0 1 0 0 0 1 0 0
     1 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 1 1 0 0 0 1 0 1 0 0 1 0 1 1 0 1 0 0 1 1 0
     0 1 0 0 1 1 1 0 0 0 0 0 1 1 0 1 0 0 0 0 0 1 0 0 0 1 0 1 0 0 1 0 1 0 0 0 0
     0 1 1 1 1 1 0 1 0 0 0]
    


```python
# Let's try something new
# fare = train_data["Fare"]; surv = train_data["Survived"]; pcl = train_data['Pclass']; 
# sex = train_data["Sex"]; sibsp = train_data["SibSp"]; parch = train_data["Parch"]; 
# emb = train_data["Embarked"]; 

train_data.corr()

X_train = pd.get_dummies(train_data[["Pclass","Sex"]])
X_test = pd.get_dummies(test_data[["Pclass","Sex"]])

# X_train_new = pd.concat([X_train,fare_train],axis = 1)
# X_test_new = pd.concat([X_test,fare_test],axis = 1)

lr = LogisticRegression()
trained_model_np = lr.fit(X_train,y)
pred = lr.predict(X_test)

my_contr_np = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': pred})
my_contr_np.to_csv('My_Contribution_NP_LessV3.csv', index= False )

print(pred)
```

    [0 1 0 0 1 0 1 0 1 0 0 0 1 0 1 1 0 0 1 1 0 0 1 0 1 0 1 0 0 0 0 0 1 1 0 0 1
     1 0 0 0 0 0 1 1 0 0 0 1 1 0 0 1 1 0 0 0 0 0 1 0 0 0 1 0 1 1 0 0 1 1 0 1 0
     1 0 0 1 0 1 0 0 0 0 0 0 1 1 1 0 1 0 1 0 0 0 1 0 1 0 1 0 0 0 1 0 0 0 0 0 0
     1 1 1 1 0 0 1 0 1 1 0 1 0 0 1 0 1 0 0 0 0 1 0 0 0 0 0 1 0 1 1 0 0 0 0 0 0
     0 0 1 0 0 1 0 0 1 1 0 1 1 0 1 0 0 1 0 0 1 1 0 0 0 0 0 1 1 0 1 1 0 0 1 0 1
     0 1 0 1 0 0 0 0 0 0 0 0 1 0 1 1 0 0 1 0 0 1 0 1 0 0 0 0 1 1 0 1 0 1 0 1 0
     1 0 1 1 0 1 0 0 0 1 0 0 0 0 0 0 1 1 1 1 0 0 0 0 1 0 1 1 1 0 0 0 0 0 0 0 1
     0 0 0 1 1 0 0 0 0 1 0 0 0 1 1 0 1 0 0 0 0 1 0 1 1 1 0 0 0 0 0 0 1 0 0 0 0
     1 0 0 0 0 0 0 0 1 1 0 0 0 1 0 0 0 1 1 1 0 0 0 0 0 0 0 0 1 0 1 0 0 0 1 0 0
     1 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 1 1 0 0 0 1 0 1 0 0 1 0 1 1 0 1 1 0 1 1 0
     0 1 0 0 1 1 1 0 0 0 0 0 1 1 0 1 0 0 0 0 0 1 0 0 0 1 0 1 0 0 1 0 1 0 0 0 0
     0 1 1 1 1 1 0 1 0 0 0]
    


```python

```


```python

```


```python

```
