import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#타이타닉 데이터 불러오기
data_titanic=pd.read_csv(r'C:\Users\82108\OneDrive\titanic dataset.csv') 
print("titanic data:\n", data_titanic.head())  # data_titanic 확인하기

#열선택 20세 이상인 사람 선택
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

grouped_missing= data_titanic.groupby('Age_category').apply(lambda x: x.isnull().sum())
print("\n child와 Adult 그룹별 결측치 확인:\n", grouped_missing)

#결측치 많은 데이터 삭제
data_titanic.drop(columns=['Cabin'],inplace=True) 

#결측치를 평균값으로 대체
data_titanic['Age'].fillna(data_titanic['Age'].mean(),inplace=True) #fillna()함수를 이용해 결측치를 평균값으로 대체함
data_titanic['Embarked'].fillna(data_titanic['Embarked'].mode()[0],inplace=True)#embarked는 평균값을 이용할 수 없으므로 최빈값 사용
print("\n결측치 처리 후:")
print(data_titanic.isnull().sum())

#age와 fare의 히스토그램
plt.figure(figsize=(14,6))

#age 히스토그램 
plt.subplot(1,2,1) #plt.subplot을 이용해 두 개의 히스토그램을 하나의 플롯에 배치함
plt.hist(data_titanic['Age'],bins=30,color='green', edgecolor='black')
plt.title('Age distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')

#fare 히스토그램
plt.subplot(1,2,2)
plt.hist(data_titanic['Fare'], bins=30, color='lightgreen', edgecolor='black')
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

#성별에 따른 생존율 산점도
plt.figure(figsize=(8,6))
sns.stripplot(x='Sex',y='Survived',data=data_titanic, jitter=True, alpha=0.6)
plt.title('Survival Rate by Gender')
plt.ylabel('Survied')
plt.xlabel('Gender')
plt.ylim(-0.1, 1.1)
plt.show()