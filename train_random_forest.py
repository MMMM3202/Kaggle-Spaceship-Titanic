from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# pd.set_option('display.max_columns',None)

# import train.csv
filename = 'database/train.csv'
data = read_csv(filename)
# print(data.groupby('Destination').size())
# print(data.dtypes)
# print(data.isnull().any())
# handling of missing values
columns = ['HomePlanet','CryoSleep','Destination','VIP']
for column in columns:
    data[column].fillna(data[column].mode()[0],inplace=True)
columns = ['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
for column in columns:
    data[column].fillna(data[column].mean(),inplace=True)
# print(data.isnull().sum())

data.drop(columns=['Cabin'],inplace=True)          #drop cabin
data = data.to_numpy()

#process data
rows = data.shape[0]
#turn homeplanet into number
for i in range(rows):
    if data[i,1] == 'Earth':
        data[i,1] = 0
    elif data[i,1] == 'Europa':
        data[i,1] = 1
    else:
        data[i,1] = 2
#turn cryosleep into number
for i in range(rows):
    if data[i,2] == True:
        data[i,2] = 1
    else:
        data[i,2] = 0
#turn destination into number
for i in range(rows):
    if data[i,3] == '55 Cancri e':
        data[i,3] = 0
    elif data[i,3] == 'PSO J318.5-22':
        data[i,3] = 1
    else:
        data[i,3] = 2
#turn vip into number
for i in range(rows):
    if data[i,6] == True:
        data[i,6] = 1
    else:
        data[i,6] = 0

#split dataset into input (x) and output (y)
x = data[1:,1:-2]
y = data[1:,-1]
x_train,x_test,y_train,y_test = train_test_split(x,y.astype('int'),test_size=0.2,random_state=0)

#create random forest classifier
random_forest = RandomForestClassifier(n_estimators=100,random_state=42)

#fit model
random_forest.fit(x_train, y_train)

#predict
y_pred = random_forest.predict(x_test)

#accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#save model
joblib.dump(random_forest, 'model/random_forest.pkl')
