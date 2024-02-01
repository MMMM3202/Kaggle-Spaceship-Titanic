import pandas as pd
from pandas import read_csv
import joblib

filename = 'database/test.csv'
data = read_csv(filename)
# save names in order to create the output file
names = data['PassengerId']
# handling of missing values
columns = ['HomePlanet','CryoSleep','Destination','VIP']
for column in columns:
    data[column].fillna(data[column].mode()[0],inplace=True)
columns = ['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
for column in columns:
    data[column].fillna(data[column].mean(),inplace=True)

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
x = data[1:,1:-1]

#load model
random_forest = joblib.load('model/random_forest.pkl')

pred = random_forest.predict(x)
pred = pd.Series(pred)
pred = pred.map({1: True, 0: False})

results = pd.DataFrame({'PassengerId': names, 'Transported': pred})
results.to_csv('result/submission.csv', index=False)