```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
col=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
df=pd.read_csv(url,names=col)

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

def pred_mod(l):
    person = np.array(l).reshape(1, -1)
    dia= pd.DataFrame(person, columns=X.columns)
    scaled= scaler.transform(dia)
    y_pred = model.predict(scaled)
    return y_pred
L=[]
for i in range(8):
    print(col[i],":",end=" ")
    l=float(input())
    L.append(l)
Y=pred_mod(L)
if(1 in Y):
    print("DIABETES")
else:
    print("NO DIABETES")
```

    Pregnancies : 

     1
    

    Glucose : 

     85
    

    BloodPressure : 

     66
    

    SkinThickness : 

     29
    

    Insulin : 

     0
    

    BMI : 

     26.6
    

    DiabetesPedigreeFunction : 

     0.351
    

    Age : 

     31
    

    NO DIABETES
    
