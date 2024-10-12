import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#타이타닉 데이터 불러오기
data1=pd.read_csv(r"C:\Users\82108\OneDrive\gender_submission.csv")
data2=pd.read_csv(r"C:\Users\82108\OneDrive\test.csv")
data3=pd.read_csv(r"C:\Users\82108\OneDrive\train.csv")
data_titanic = pd.concat([data1, data2, data3], ignore_index=True)
print("titanic data:\n", data_titanic.head())  # data_titanic 확인하기

#열선택 
data_titanic.loc[:,['Sex','Age','Survived','SibSp','Embarked']].head() 

#나이가 20세 이상인 사람 선택
person_over_20=data_titanic[data_titanic['Age']>=20] 
print("나이가 20살 이상인 사람:\n", person_over_20)

#survived한 사람 선택
person_survived=data_titanic[data_titanic['Survived']==1] 
print("살아남은 사람:\n", person_survived)

#queenstown에서 승선한 사람 선택
person_embarked =data_titanic[data_titanic['Embarked']=='Q'] 
print("Queenstown에서 승선한 사람:\n", person_embarked)

#새로운 열 추가하기
bins=[0,19,100] # 나이 구간 설정(0~19:child, 20~100:adult)
labels=['Child','Adult']
data_titanic['Age_category']=pd.cut(data_titanic['Age'],bins=bins,labels=labels,right=True )
print(data_titanic[['Age','Age_category']].head(10)) #10번째 행까지 확인

#결측치 확인
print("각 열의 결측치 확인:")
print(data_titanic.isnull().sum()) #각 열에서의 결측치 확인

#결측치 많은 데이터 삭제
data_titanic.drop(columns=['Cabin'],inplace=True) 

#결측치를 평균값으로 대체
data_titanic['Survived'].fillna(data_titanic['Survived'].mean(),inplace=True)
data_titanic['Pclass'].fillna(data_titanic['Pclass'].mean(),inplace=True)
data_titanic['Name'].fillna(data_titanic['Name'].mode()[0],inplace=True)
data_titanic['Age'].fillna(data_titanic['Age'].mean(),inplace=True)
data_titanic['SibSp'].fillna(data_titanic['SibSp'].mean(),inplace=True)
data_titanic['Ticket'].fillna(data_titanic['Ticket'].mode()[0],inplace=True)
data_titanic['Fare'].fillna(data_titanic['Fare'].mean(),inplace=True)
data_titanic['Parch'].fillna(data_titanic['Parch'].mean(),inplace=True) #fillna()함수를 이용해 결측치를 평균값으로 대체함
data_titanic['Embarked'].fillna(data_titanic['Embarked'].mode()[0],inplace=True)
print(data_titanic.isnull().sum())

#age와 fare의 히스토그램
plt.figure(figsize=(15,6))

#age 히스토그램 
plt.subplot(1,2,1) #plt.subplot을 이용해 두 개의 히스토그램을 하나의 플롯에 배치함
plt.hist(data_titanic['Age'],bins=40,color='green', edgecolor='black')
plt.title('Age distribution',fontsize=15)
plt.xlabel('Age')
plt.ylabel('count')

#fare 히스토그램
plt.subplot(1,2,2)
plt.hist(data_titanic['Fare'], bins=40, color='lightgreen', edgecolor='black')
plt.title('Fare Distribution',fontsize=15)
plt.xlabel('Fare')
plt.ylabel('count')
plt.tight_layout()
plt.show()

#성별에 따른 생존율 산점도
survival_rate_by_gender = data_titanic.groupby('Sex')['Survived'].mean().reset_index()
sns.scatterplot(x='Sex', y='Survived', data=survival_rate_by_gender,s=60, color='blue')
plt.title('Survival Rate by Gender', fontsize=15) 
plt.ylabel('Survival Rate')
plt.xlabel('Gender')
plt.ylim(0,1)
plt.show()