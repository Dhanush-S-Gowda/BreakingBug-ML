import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from yellowbrick.cluster import KElbowVisualizer
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Correct import
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# Corrected Wrong lib
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB

#corected
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report , mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("Kaggle/input/heart-disease-data/heart_disease_uci.csv")

df.head()

df.info()

df.shape

df['id'].min(), df['id'].max()

df['age'].min(), df['age'].max()

df['age'].describe()

import seaborn as sns

custom_colors = ["#FF5733", "#3366FF", "#33FF57"]

sns.histplot(df['age'], kde=True, color="#FF5733", palette=custom_colors)

sns.histplot(df['age'], kde=True)
plt.axvline(df['age'].mean(), color='Red')
plt.axvline(df['age'].median(), color= 'Green')
plt.axvline(df['age'].mode()[0], color='Blue')

print('Mean', df['age'].mean())
print('Median', df['age'].median())
print('Mode', df['age'].mode())

fig = px.histogram(data_frame=df, x='age', color= 'sex')
fig.show()

df['sex'].value_counts()

male_count = 726
female_count = 194

total_count = male_count + female_count

male_percentage = (male_count/total_count)*100
female_percentages = (female_count/total_count)*100

print(f'Male percentage i the data: {male_percentage:.2f}%')
print(f'Female percentage in the data : {female_percentages:.2f}%')

difference_percentage = ((male_count - female_count)/female_count) * 100
print(f'Males are {difference_percentage:.2f}% more than female in the data.')

726/194

df.groupby('sex')['age'].value_counts()

df['dataset'].value_counts() # Corrected 'dataset' spelling and changed to correct method

fig =px.bar(df, x='dataset', color='sex')
fig.show()

print (df.groupby('sex')['dataset'].value_counts())

fig = px.histogram(data_frame=df, x='age', color= 'dataset')
fig.show()

print("___________________________________________________________")
print("Mean of the dataset: ", df['age'].mean()) # corrected
print("___________________________________________________________")
print("Median of the dataset: ", df['age'].median()) # corrected
print("___________________________________________________________")
print("Mode of the dataset: ", df['age'].mode()) # corrected
print("___________________________________________________________")

df['cp'].value_counts()

sns.countplot(df, x='cp', hue= 'sex')

sns.countplot(df,x='cp',hue='dataset')

fig = px.histogram(data_frame=df, x='age', color='cp')
fig.show()

df['trestbps'].describe()

print(f"Percentage of missing values in trestbps column: {df['trestbps'].isnull().sum() /len(df) *100:.2f}%")

imputer1 = IterativeImputer(max_iter=10, random_state=42)

imputer1.fit(df[['trestbps']])

df['trestbps'] = imputer1.transform(df[['trestbps']])

print(f"Missing values in trestbps column: {df['trestbps'].isnull().sum()}")

df.info()

imputer2 = IterativeImputer(max_iter=10, random_state=42)

#corrected
imputer2.fit(df[['ca', 'oldpeak', 'chol', 'thalch']])
df[['ca', 'oldpeak', 'chol', 'thalch']] = imputer2.transform(df[['ca', 'oldpeak', 'chol', 'thalch']])

(df.isnull().sum()/ len(df)* 100).sort_values(ascending=False)

print(f"The missing values in thal column are: {df['thal'].isnull().sum()}")

df['thal'].value_counts()

df.tail()

df.isnull().sum()[df.isnull().sum() > 0].sort_values(ascending=True) #corrected

missing_data_cols = df.isnull().sum()[df.isnull().sum()>0].index.tolist()

missing_data_cols

cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols

Num_cols = df.select_dtypes(exclude='object').columns.tolist()
Num_cols

print(f'categorical Columns: {cat_cols}')
print(f'numerical Columns: {Num_cols}')

categorical_cols = ['sex', 'dataset', 'cp', 'restecg', 'slope', 'thal']
bool_cols = ['fbs', 'exang']
numerical_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca', 'num']

def impute_categorical_missing_data(passed_col):
  df_null = df[df[passed_col].isnull()]
  df_not_null = df[df[passed_col].notnull()]

  X = df_not_null.drop(passed_col, axis=1)
  y = df_not_null[passed_col]

  other_missing_cols = [col for col in missing_data_cols if col != passed_col]

  encoders = {}
  for col in X.columns:
      if X[col].dtype == 'object':
          encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
          X_encoded = encoder.fit_transform(X[[col]])
          X = X.drop(col, axis=1)
          X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out([col]), index=X.index)
          X = pd.concat([X, X_encoded_df], axis=1)
          encoders[col] = encoder

  label_encoder = LabelEncoder()
  if passed_col in bool_cols:
      y = label_encoder.fit_transform(y)

  imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=16), add_indicator=True)
  imputed_values = imputer.fit_transform(X)
  imputed_values = pd.DataFrame(imputed_values, columns=X.columns, index=X.index)

  X_train, X_test, y_train, y_test = train_test_split(imputed_values, y, test_size=0.2, random_state=42)
  rf_classifier = RandomForestClassifier(random_state=16)
  rf_classifier.fit(X_train, y_train)

  y_pred = rf_classifier.predict(X_test)
  acc_score = accuracy_score(y_test, y_pred)
  print(f"The feature '{passed_col}' has been imputed with {round((acc_score * 100), 2)}% accuracy\n")

  X_null = df_null.drop(passed_col, axis=1)

  for col in X_null.columns:
      if X_null[col].dtype == 'object':
          encoder = encoders.get(col)
          if encoder:
              X_null_encoded = encoder.transform(X_null[[col]])
              X_null = X_null.drop(col, axis=1)
              X_null_encoded_df = pd.DataFrame(X_null_encoded, columns=encoder.get_feature_names_out([col]), index=X_null.index)
              X_null = pd.concat([X_null, X_null_encoded_df], axis=1)

  X_null = X_null.reindex(columns=imputed_values.columns, fill_value=0)

  imputed_null_values = imputer.transform(X_null)
  imputed_null_values = pd.DataFrame(imputed_null_values, columns=X.columns, index=X_null.index)

  if len(df_null) > 0:
      predictions = rf_classifier.predict(imputed_null_values)
      df.loc[df[passed_col].isnull(), passed_col] = predictions
      if passed_col in bool_cols:
          df[passed_col] = df[passed_col].map({0: False, 1: True})

  df_combined = pd.concat([df_not_null, df_null])

  print(df_combined[passed_col])

  return df_combined[passed_col]

def impute_continuous_missing_data(passed_col):
    df_null = df[df[passed_col].isnull()]
    df_not_null = df[df[passed_col].notnull()]

    X = df_not_null.drop(passed_col, axis=1)
    y = df_not_null[passed_col]

    other_missing_cols = [col for col in missing_data_cols if col != passed_col]

    encoders = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            X_encoded = encoder.fit_transform(X[[col]])
            X = X.drop(col, axis=1)
            X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out([col]), index=X.index)
            X = pd.concat([X, X_encoded_df], axis=1)
            encoders[col] = encoder

    imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=16), add_indicator=True)
    imputed_values = imputer.fit_transform(X)
    imputed_values = pd.DataFrame(imputed_values, columns=X.columns, index=X.index)

    X_train, X_test, y_train, y_test = train_test_split(imputed_values, y, test_size=0.2, random_state=42)
    rf_regressor = RandomForestRegressor(random_state=16)
    rf_regressor.fit(X_train, y_train)

    y_pred = rf_regressor.predict(X_test)
    print("MAE =", mean_absolute_error(y_test, y_pred))
    print("RMSE =", mean_squared_error(y_test, y_pred, squared=False))
    print("R2 =", r2_score(y_test, y_pred))

    X_null = df_null.drop(passed_col, axis=1)

    for col in X_null.columns:
        if X_null[col].dtype == 'object':
            encoder = encoders.get(col)
            if encoder:
                X_null_encoded = encoder.transform(X_null[[col]])
                X_null = X_null.drop(col, axis=1)
                X_null_encoded_df = pd.DataFrame(X_null_encoded, columns=encoder.get_feature_names_out([col]), index=X_null.index)
                X_null = pd.concat([X_null, X_null_encoded_df], axis=1)

    X_null = X_null.reindex(columns=imputed_values.columns, fill_value=0)

    imputed_null_values = imputer.transform(X_null)
    imputed_null_values = pd.DataFrame(imputed_null_values, columns=X.columns, index=X_null.index)

    if len(df_null) > 0:
        predictions = rf_regressor.predict(imputed_null_values)
        df.loc[df[passed_col].isnull(), passed_col] = predictions

    df_combined = pd.concat([df_not_null, df_null])

    return df_combined[passed_col]

df.isnull().sum().sort_values(ascending=False)

import warnings
warnings.filterwarnings('ignore')

for col in missing_data_cols:
    print("Missing Values", col, ":", str(round((df[col].isnull().sum() / len(df)) * 100, 2))+"%")
    if col in categorical_cols:
        df[col] = impute_categorical_missing_data(col)
    elif col in numerical_cols:
        df[col] = impute_continuous_missing_data(col)
    else:
        pass

df.isnull().sum()

print("_________________________________________________________________________________________________________________________________________________")

sns.set(rc={"axes.facecolor":"#87CEEB","figure.facecolor":"#EEE8AA"})  # Change figure background color

palette = ["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"]
cmap = ListedColormap(["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"])

plt.figure(figsize=(10,8))

for i, col in enumerate(palette):
    plt.subplot(3,2)
    sns.boxenplot(color=palette[i % len(palette)])  # Use modulo to cycle through colors
    plt.title(i)

plt.show()

df[df['trestbps']==0]

df= df[df['trestbps']!=0]

sns.set(rc={"axes.facecolor":"#B76E79","figure.facecolor":"#C0C0C0"})
modified_palette = ["#C44D53", "#B76E79", "#DDA4A5", "#B3BCC4", "#A2867E", "#F3AB60"]
cmap = ListedColormap(modified_palette)

plt.figure(figsize=(10,8))

for i, col in enumerate(cols):
    plt.subplot(3,2)
    sns.boxenplot( color=palette[i % len(palette)])  # Use modulo to cycle through colors
    plt.title(col)

plt.show()

df.trestbps.describe()

df.describe()

print("___________________________________________________________________________________________________________________________________________________________________")

sns.set(rc={"axes.facecolor": "#FFF9ED", "figure.facecolor": "#FFF9ED"})

night_vision_palette = ["#00FF00", "#FF00FF", "#00FFFF", "#FFFF00", "#FF0000", "#0000FF"]

plt.figure(figsize=(10, 8))
for i, col in enumerate(cols):
    plt.subplot(3,2)
    sns.boxenplot( color=palette[i % len(palette)])  # Use modulo to cycle through colors
    plt.title(col)

plt.show()

df.age.describe()

palette = ["#999999", "#666666", "#333333"]

sns.histplot(data=df,
             x='trestbps',
             kde=True,
             color=palette[0])

plt.title('Resting Blood Pressure')
plt.xlabel('Pressure (mmHg)')
plt.ylabel('Count')

plt.style.use('default')
plt.rcParams['figure.facecolor'] = palette[1]
plt.rcParams['axes.facecolor'] = palette[2]

sns.histplot(df, x='trestbps', kde=True, palette = "Spectral", hue ='sex')

df.info()

df.columns

df.head()

X= df.drop('num', axis=1)
y = df['num']

"""encode X data using separate label encoder for all categorical columns and save it for inverse transform"""
#corrected
for col in X.columns:
    if X[col].dtype == 'object':
        label_encoder = LabelEncoder()
        X[col] = label_encoder.fit_transform(X[col].astype(str))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# changed
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestRegressor, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings('ignore')

#changed
models = [
    ('Logistic Regression', LogisticRegression(random_state=42)),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=42)),
    ('KNeighbors Classifier', KNeighborsClassifier()),
    ('Decision Tree Classifier', DecisionTreeClassifier(random_state=42)),
    ('AdaBoost Classifier', AdaBoostClassifier(random_state=42)),
    ('Random Forest', RandomForestRegressor(random_state=42)),
    ('XGBoost Classifier', XGBClassifier(random_state=42)),
    ('Support Vector Machine', SVC(random_state=42)),
    ('Naive Bayes Classifier', GaussianNB())
]

best_model = None
best_accuracy = 0.0

#changed
for name, model in models:
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False)),
        ('model', model)
    ])

    scores = cross_val_score(pipeline, X_train, y_train, cv=5)

    mean_accuracy = scores.mean()

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    y_test = y_test.astype(int)
    y_pred = y_pred.astype(int)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model: {name}")
    print(f"Cross Validation Accuracy: {mean_accuracy:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}\n")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = pipeline

print("Best Model: ", best_model)

categorical_cols = ['sex', 'dataset', 'cp', 'restecg', 'slope', 'thal'] #changed

#changed
def evaluate_classification_models(X, y, categorical_columns):
    X_encoded = X.copy()
    encoders = {}

    for col in categorical_columns:
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        X_encoded_cat = encoder.fit_transform(X_encoded[[col]])
        X_encoded = X_encoded.drop(col, axis=1)
        X_encoded_cat_df = pd.DataFrame(X_encoded_cat, columns=encoder.get_feature_names_out([col]), index=X_encoded.index)
        X_encoded = pd.concat([X_encoded, X_encoded_cat_df], axis=1)
        encoders[col] = encoder

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "KNN": KNeighborsClassifier(),
        "NB": GaussianNB(),
        "SVM": SVC(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": xgb.XGBClassifier(random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42)
    }

    results = {}
    best_model = None
    best_accuracy = 0.0

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = name

    return results, best_model

results, best_model = evaluate_classification_models(X, y, categorical_cols)
print("Model accuracies:", results)
print("Best model:", best_model)

X = df[categorical_cols]
y = df['num']

#chnged
def hyperparameter_tuning(X, y, categorical_columns, models):
    results = {}

    X_encoded = X.copy()
    onehotencoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    for col in categorical_columns:
        X_encoded = X_encoded.join(pd.DataFrame(onehotencoder.fit_transform(X_encoded[[col]]),
                                                columns=onehotencoder.get_feature_names_out([col]),
                                                index=X_encoded.index))
        X_encoded = X_encoded.drop(col, axis=1)

    X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    for model_name, model in models.items():
        param_grid = {}
        if model_name == 'Logistic Regression':
            param_grid = {'C': [0.1, 1, 10, 100]}
        elif model_name == 'KNN':
            param_grid = {'n_neighbors': [3, 5, 7, 9]}
        elif model_name == 'NB':
            param_grid = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]}
        elif model_name == 'SVM':
            param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 1, 10, 100]}
        elif model_name == 'Decision Tree':
            param_grid = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
        elif model_name == 'Random Forest':
            param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
        elif model_name == 'XGBoost':
            param_grid = {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7]}
        elif model_name == 'GradientBoosting':
            param_grid = {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7]}
        elif model_name == 'AdaBoost':
            param_grid = {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [50, 100, 200]}

        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)

        results[model_name] = {'best_params': best_params, 'accuracy': accuracy}

    return results

#changed
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "NB": GaussianNB(),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": xgb.XGBClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier()
}

results = hyperparameter_tuning(X, y, categorical_cols, models)
for model_name, result in results.items():
    print("Model:", model_name)
    print("Best hyperparameters:", result['best_params'])
    print("Accuracy:", result['accuracy'])
    print()
