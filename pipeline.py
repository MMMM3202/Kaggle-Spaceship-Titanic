import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

sns.set(style='darkgrid', font_scale=2)
import warnings

warnings.filterwarnings('ignore')

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Models
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn import set_config

df_train = pd.read_csv('database/train.csv')
df_test = pd.read_csv('database/test.csv')

df_train[['Deck', 'Num', 'Side']] = df_train['Cabin'].str.split('/', expand=True)
df_test[['Deck', 'Num', 'Side']] = df_test.Cabin.str.split('/', expand=True)

df_train['total_spent'] = df_train['RoomService'] + df_train['FoodCourt'] + df_train['ShoppingMall'] + df_train['Spa'] + \
                          df_train['VRDeck']
df_test['total_spent'] = df_test['RoomService'] + df_test['FoodCourt'] + df_test['ShoppingMall'] + df_test['Spa'] + \
                         df_test['VRDeck']

sns.set_style('whitegrid')
# fig1, ax1 = plt.subplots()
# df_train['Age'].hist(ax=ax1, bins=30)
# plt.show()

df_train['AgeGroup'] = 0
for i in range(7):
    df_train.loc[(df_train.Age >= 10 * i) & (df_train.Age < 10 * (i + 1)), 'AgeGroup'] = i
# Same for test data
df_test['AgeGroup'] = 0
for i in range(7):
    df_test.loc[(df_test.Age >= 10 * i) & (df_test.Age < 10 * (i + 1)), 'AgeGroup'] = i
# plt.figure(figsize=(10, 6))
# sns.countplot(x='AgeGroup', hue='Transported', data=df_train)
# plt.show()
X = df_train.drop('Transported', axis=1)
X = X.drop(['PassengerId', 'Name'], axis=1)
y = df_train['Transported']
X['Num'] = pd.to_numeric(X['Num'])

cat_cols = X.select_dtypes('object').columns.to_list()
num_cols = X.select_dtypes(exclude='object').columns.to_list()

# Making Seperate Preprocessing Pipelines for numeric and categorical columns
numeric_preprocessor = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),
                                       ('scaler', StandardScaler())])
categorical_preprocessor = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore')),
                                           ('imputer', SimpleImputer(strategy='constant')), ])
preprocessor = ColumnTransformer([
    ('categorical', categorical_preprocessor, cat_cols),
    ('numeric', numeric_preprocessor, num_cols)])

Pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', GradientBoostingClassifier())])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=1) \

Pipe.fit(X_train, y_train)
pred = Pipe.predict(X_train)
pred_y = Pipe.predict(X_val)
print('Train accuracy ', accuracy_score(y_train.values, pred))
print('Validation accuracy', accuracy_score(y_val.values, pred_y))

param_grid = {'model__n_estimators': [500, 1000], 'model__learning_rate': [0.1, 0.2], 'model__verbose': [1],
              'model__max_depth': [2, 3]}
from sklearn.model_selection import GridSearchCV

#create  a grid search object
gcv = GridSearchCV(Pipe, param_grid=param_grid, cv=5, scoring="roc_auc", verbose=0)
try:
    with open('gcv.pkl', 'rb') as gcv_paremeters:
        gcv = joblib.load(gcv_paremeters)
except FileNotFoundError:
    gcv.fit(X, y)
    joblib.dump(gcv, 'gcv.pkl')

pred_gcv = gcv.predict(X_train)
pred_y_gcv = gcv.predict(X_val)
print('Train accuracy ', accuracy_score(y_train.values, pred_gcv))
print('Validation accuracy', accuracy_score(y_val.values, pred_y_gcv))
y_pred = gcv.predict(df_test)

sub = pd.DataFrame({'Transported': y_pred.astype(bool)}, index=df_test['PassengerId'])
sub.to_csv('submission.csv.csv')
