import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score
from scipy.stats import ttest_ind

# Load data
df = pd.read_csv('train.csv')

# Dự đoán Age bằng Linear Regression
age_df = df[['Age', 'Pclass', 'Sex', 'SibSp', 'Parch']].copy()
age_df['Sex'] = age_df['Sex'].map({'male': 0, 'female': 1})
known_age = age_df[age_df['Age'].notnull()]
unknown_age = age_df[age_df['Age'].isnull()]

lr = LinearRegression()
lr.fit(known_age.drop(columns='Age'), known_age['Age'])
predicted_ages = lr.predict(unknown_age.drop(columns='Age'))
df.loc[df['Age'].isnull(), 'Age'] = predicted_ages

# Điền embarked bằng KNN
knn_df = df[['Embarked', 'Pclass', 'Fare', 'Sex']].copy()
knn_df['Sex'] = knn_df['Sex'].map({'male': 0, 'female': 1})
embarked_map = {'S': 0, 'C': 1, 'Q': 2}
knn_df['Embarked'] = knn_df['Embarked'].map(embarked_map)

imputer = KNNImputer(n_neighbors=3)
knn_imputed = imputer.fit_transform(knn_df)
df['Embarked'] = pd.Series(knn_imputed[:,0]).round().map({0: 'S', 1: 'C', 2: 'Q'})

# Trích xuất Deck từ Cabin
df['Deck'] = df['Cabin'].apply(lambda x: x[0] if pd.notnull(x) else 'Unknown')

# Tạo family_size và is_alone
df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
df['Is_Alone'] = (df['Family_Size'] == 1).astype(int)

# Trích xuất Title từ Name
df['Title'] = df['Name'].str.extract(r',\s*([^\.]*)\s*\.', expand=False)
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',
                                   'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

# Tạo fare_per_person
df['Fare_Per_Person'] = df['Fare'] / df['Family_Size']

# Boxplot fare_per_person theo Pclass và Survived
sns.boxplot(data=df, x='Pclass', y='Fare_Per_Person', hue='Survived')
plt.title('Fare per Person by Pclass and Survival')
plt.show()

# T-test giữa nhóm sống và không sống
survived = df[df['Survived'] == 1]['Fare_Per_Person']
not_survived = df[df['Survived'] == 0]['Fare_Per_Person']
t_stat, p_val = ttest_ind(survived, not_survived)
print(f'T-test: t={t_stat:.3f}, p={p_val:.5f}')

# Chuẩn bị pipeline và dữ liệu
categorical_cols = ['Sex', 'Embarked', 'Deck', 'Title']
numeric_cols = ['Age', 'Fare_Per_Person', 'Family_Size', 'Is_Alone', 'Pclass']

X = df[categorical_cols + numeric_cols]
y = df['Survived']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [4, 6, 8, None]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid.fit(X, y)
print(f'Best Params: {grid.best_params_}')

# Đánh giá mô hình
y_pred = cross_val_predict(grid.best_estimator_, X, y, cv=5, method='predict')
y_proba = cross_val_predict(grid.best_estimator_, X, y, cv=5, method='predict_proba')[:,1]
print(classification_report(y, y_pred))
print('ROC AUC:', roc_auc_score(y, y_proba))

# Feature Importance
importances = grid.best_estimator_.named_steps['classifier'].feature_importances_
feature_names = grid.best_estimator_.named_steps['preprocessor'].get_feature_names_out()
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=feat_imp.values[:10], y=feat_imp.index[:10])
plt.title("Top 10 Feature Importances")
plt.show()
