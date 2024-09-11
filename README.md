# EXNO2DS
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
## STEP1:Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.
```
import pandas as pd
data=pd.read_csv("/content/titanic_dataset (1).csv")
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/9ce4a508-0fc7-4123-af4f-8d6a540bd563)
## STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.
```
from sklearn.impute import SimpleImputer
imputer_age = SimpleImputer(strategy='median')
data['Age'] = imputer_age.fit_transform(data[['Age']])
imputer_fare = SimpleImputer(strategy='mean')
data['Fare'] = imputer_fare.fit_transform(data[['Fare']])
imputer_embarked = SimpleImputer(strategy='most_frequent')
data['Embarked'] = imputer_embarked.fit_transform(data[['Embarked']]).ravel()
data = data.drop(columns=['Cabin'])
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/8b6efae3-98af-41f5-b910-6d6509cafc14)
## STEP 3: Use boxplot method to analyze the outliers of the given dataset.
```
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.boxplot(data=data, x='Age')
plt.title('Boxplot of Age')
plt.subplot(1,2,2)
sns.boxplot(data=data, x='Fare')
plt.title('Boxplot of Fare')
plt.show()
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/f4f0ed21-84fb-4571-bdf3-43c670267065)
## STEP 4: Remove the outliers using Inter Quantile Range method.
```
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers in Age and Fare
data = remove_outliers(data, 'Age')
data = remove_outliers(data, 'Fare')
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/9de0c115-cc0d-4465-84ba-d55d7c659c4c)
## STEP 5: Use Countplot method to analyze in a graphical method for categorical data.
```
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.countplot(data=data, x='Sex')
plt.title('Countplot of Sex')

plt.subplot(1,2,2)
sns.countplot(data=data, x='Embarked')
plt.title('Countplot of Embarked')

plt.show()
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/f6120622-11d9-4245-85c9-cda22762f7aa)
## STEP 6: Use displot method to represent the univariate distribution of data.
```
plt.figure(figsize=(10,5))
sns.displot(data['Age'], kde=True)
plt.title('Distribution of Age')
plt.show()

sns.displot(data['Fare'], kde=True)
plt.title('Distribution of Fare')
plt.show()
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/dda9b042-70b3-4be8-b0fd-904c5790fbfc)
## STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.
```
cross_tab_sex_survived = pd.crosstab(data['Sex'], data['Survived'])
print(cross_tab_sex_survived)
cross_tab_sex_survived.plot(kind='bar', figsize=(10,5), stacked=True)
plt.title('Survival by Gender')
plt.show()
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/7914c647-612f-4259-975b-a7cf884660ea)
## STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.
```
plt.figure(figsize=(8,6))
correlation_matrix = data[['Age', 'Fare', 'SibSp', 'Parch']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap of Numerical Features')
plt.show()
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/99377b4d-3903-44e1-b1ed-68be6298b5f9)
# RESULT
        <<INCLUDE YOUR RESULT HERE>>
