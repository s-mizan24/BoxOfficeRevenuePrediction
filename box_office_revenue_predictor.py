def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


"""#### Importing Libraries"""

import ast
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_log_error
import matplotlib
matplotlib.use('Agg')
"""#### Loading dataset

"""

main_train = pd.read_csv('train.csv')
main_test = pd.read_csv('test.csv')
# sample_submission = pd.read_csv('sample_submission.csv')


train = main_train.copy(deep=True)

# submission = main_train[["id", "revenue"]]
# submission.to_csv('sample_submission.csv', index=False)

test = main_test.copy(deep=True)



trains = main_train.copy(deep=True)
tests = main_test.copy(deep=True)

"""### Handling missing budget data in 'trains' for observational purposes"""


"""### Pre-processing and EDA"""

train.drop(train[train['revenue']<1000].index, inplace=True)
train.reset_index(drop=True, inplace=True)

"""#### get_names()
Inputs: list of dictionaries is string format

Returns: List of names of desired attribute
"""

budget_low = trains["budget"].quantile(0.03)
budget_high = trains["budget"].quantile(0.97)

filtered_budget_trains = trains[(trains["budget"] < budget_high) & (trains["budget"] > budget_low)]
budget_mean = int(filtered_budget_trains["budget"].mean())


trains['budget'].replace(0, budget_mean, inplace=True)

g = sns.countplot(data=train, x="revenue")
g.set(xlim = (0, 100000000))
def format_func(value, tick_number):
  return value/1000000+'M'
g.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

"""Handling 'belongs_top_collection'"""
####IMG

ast.literal_eval(train['belongs_to_collection'][0])

train['has_collection'] = train['belongs_to_collection'].apply(lambda x: 1 if isinstance(x, str) else 0)

sns.catplot(x = 'has_collection', y = 'revenue', data=train, height=7, aspect=0.8)
####IMG

train['collection_names'] = train['belongs_to_collection'].fillna('[]').map(eval).map(lambda x: [collection['name'] for collection in x])

train['collection_names'].map(lambda x: 1 if x else 0)

"""Handling 'homepage'

Majority of null values are for the columns, 'belongs_to_collection' and 'homepage'.

Let us see if having a homepage is significant or not?
"""

train['has_homepage'] = train['homepage'].notnull().astype('int')

sns.set_theme(style="ticks")
# sns.catplot(x='has_homepage', y='revenue', data=train, height=7, aspect=0.8)
####IMG

"""It seems, having a homepage has its significance when it comes to revenues

Handling 'budget'

There are no none values but let us see if there are any anomalies in this column.
"""

sns.set(rc = {'figure.figsize':(15,8)})
sns.set_style("whitegrid", {'axes.grid' : False})
# sns.scatterplot(x='budget', y='revenue', data=train)
####IMG


"""Handling 'genres'"""

train['genres'] = train['genres'].fillna('[]').map(eval).map(lambda x: [g['name'].lower() for g in x])

train['genres'].map(lambda x: [g for g in x][:2])

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
genre_mlb = mlb.fit_transform(train['genres'])
genre_labels = mlb.classes_
genre_df = pd.DataFrame(genre_mlb, columns=genre_labels)

import pickle

with open('genre_mlb.pkl', 'wb') as f:
  pickle.dump(mlb, f)

test_genres = pd.DataFrame({"genre": [["action", "Drama","Fantasy"], ["TEST1","Action"], ["Drama"], ["sci-fi", "Drama"]]})

with open('genre_mlb.pkl', 'rb') as file:
  genre_mlb = pickle.load(file)
test_mlb = genre_mlb.transform(test_genres['genre'])
test_genre_df = pd.DataFrame(test_mlb, columns=genre_labels)


"""Analysing original_language

We can see that 'en' (English) as an original language for a movie has a significant affect on revenue.


"""

plt.subplots(figsize=(12,8))
# sns.boxplot(x=train['original_language'], y=train['revenue'])
####IMG

train['is_en'] = train['original_language'].map(lambda x: 1 if x == 'en' else 0)
"""Handling 'popularity'

No null values
"""

plt.figure(figsize=(14, 6))
# sns.scatterplot(x='popularity', y='revenue', data=train[train['popularity']<50])
####IMG

"""Handling 'runtime'"""

train['runtime'].fillna(value=train['runtime'].median(), inplace=True)

plt.figure(figsize=(12,8))
# sns.scatterplot(x='runtime', y='revenue', data=train)
####IMG

"""Handling 'production_countries' and 'production companies'"""

train['production_companies'] = train['production_companies'].fillna('[]').map(eval).map(lambda x: [g['name'].lower() for g in x])

train['production_countries'] = train['production_countries'].fillna('[]').map(eval).map(lambda x: [g['name'].lower() for g in x])

prod_companies = pd.DataFrame(train['production_companies'].apply(lambda x: pd.Series(x)).stack().value_counts()).reset_index()
prod_companies.columns= ['companies', 'movies_count']

sns.set_theme(style = "darkgrid")
sns.set(rc={'figure.figsize':(18, 14)})
# sns.barplot(y="companies", x="movies_count", data=prod_companies.head(50))
####IMG

train['num_of_companies'] = train['production_companies'].map(len)

sns.set(rc = {'figure.figsize':(15,8)})
sns.set_style("whitegrid", {'axes.grid' : False})
# sns.catplot(x='num_of_companies', y='revenue', data=train, height=7, aspect=0.9)
####IMG

prod_countries = pd.DataFrame(train['production_countries'].apply(lambda x: pd.Series(x)).stack().value_counts()).reset_index()
prod_countries.columns= ['countries', 'movies_count']

sns.set_theme(style = "darkgrid")
sns.set(rc={'figure.figsize':(20, 8)})
# sns.barplot(y="countries", x="movies_count", data=prod_countries.head(50))
####IMG


train['usa_produced'] = train['production_countries'].apply(lambda x: 1 if 'united states of america' in x else 0)

# sns.catplot(x='usa_produced', y='revenue', data=train, height=7, aspect=0.9)
####IMG

"""Handling 'status'"""

train['is_released'] = train['status'].apply(lambda x: 1 if x=='Released' else 0)
# sns.catplot(x='is_released', y='revenue', data=train, height=7, aspect=0.9)
####IMG

"""Extracting 'release year' and 'release month' as they can be important information"""

train['release_date'] = train['release_date'].map(pd.to_datetime)

train['release_year'] = train['release_date'].apply(lambda x: x.year)

year = train['release_year']
train['release_year'] = np.where(year>2017, year - 100, year)

# sns.catplot(x='budget', y='release_year', data=train, orient="h", height=14, aspect=0.7)
####IMG

train['release_month'] = train['release_date'].apply(lambda x: x.month)

# sns.catplot(y='revenue', x='release_month', data=train, orient="v", height=8, aspect=1.3)
####IMG

"""Handling 'spoken_languages'"""

train['spoken_languages'] = train['spoken_languages'].fillna('[]').map(eval).map(lambda x: [g['name'].lower() for g in x])

train['num_of_spoken_languages'] = train['spoken_languages'].map(len)

# sns.catplot(x='num_of_spoken_languages', y='revenue', data=train, height=8)
####IMG

"""Dropping all unwanted columns from the train dataset"""

train.drop(['id', 'imdb_id', 'overview', 'poster_path', 'title', 'tagline', 'Keywords', 'belongs_to_collection', 'collection_names', 'homepage', 'original_title', 'original_language', 'production_companies', 'production_countries', 'release_date', 'spoken_languages', 'status', 'cast', 'crew'], axis=1, inplace=True)

"""### Encoding Genres
"""

mlb = MultiLabelBinarizer()
genre_mlb = mlb.fit_transform(train['genres'])
genre_labels = mlb.classes_
genre_df = pd.DataFrame(genre_mlb, columns=genre_labels)

train = train.join(genre_df)

train.drop(['genres'], axis=1, inplace=True)

train[train.columns.difference(['revenue'])]


def inverse_log(x):
  return np.exp(x)-1

trainX = train.copy(deep=True)

"""Saving Preprocessed features to 'processed_train.csv'"""

trainX.to_csv('processed_train.csv', index=False)

trainX[trainX.budget==0]['revenue'].unique()

from sklearn.preprocessing import PolynomialFeatures

predictors = ['popularity', 'revenue', 'has_homepage',
             'num_of_companies', 'usa_produced', 'action', 'adventure']
lr_budget = RandomForestRegressor(max_depth=8, min_samples_split=10)
traindf = trainX[trainX.budget!=0]
traindf = np.log1p(traindf)
y = traindf.budget
lr_budget.fit(traindf[predictors], y)

indices = [list(trainX.columns).index(key) for key in predictors]
trainX['budget'] = trainX.apply(lambda row: row[0] if row[0] else inverse_log(lr_budget.predict([np.log1p(row[indices])])[0]), axis=1)


trainX['log_budget'] = np.log1p(trainX['budget'])

sns.set(rc = {'figure.figsize':(15,8)})
sns.set_style("whitegrid", {'axes.grid' : False})
# g = sns.scatterplot(x='budget', y='revenue', data=trainX)
# g.set(title="Budget-Revenue graph before Log Transformation", xlabel="Budget", ylabel="Revenue")
####IMG

trainX['log_revenue'] = np.log1p(trainX['revenue'])

sns.set(rc = {'figure.figsize':(15,8)})
sns.set_style("whitegrid", {'axes.grid' : False})
# g = sns.scatterplot(x='log_budget', y='log_revenue', data=trainX)
# g.set(title="Budget-Revenue graph after Log Transformation", xlabel="log(Budget)", ylabel="log(Revenue)")
####IMG



# g = sns.displot(data=trainX, x="budget", height=8, aspect=1.8, binwidth=10000000)
# g.set(title="Budget Distribution before Log Transformation", xlabel="Budget", ylabel="Count")
####IMG

# g = sns.displot(data=trainX, x="log_budget", height=8, aspect=1.8)
# g.set(title="Budget Distribution after Log Transformation", xlabel="log(Budget)", ylabel="Count")
####IMG

# g = sns.displot(data=trainX, x="revenue", height=8, aspect=1.8, binwidth=10000000)
# g.set(title="Revenue Distribution before Log Transformation", xlabel="Revenue", ylabel="Count", xlim=(0,600000000))
####IMG

# g = sns.displot(data=trainX, x="log_revenue", height=8, aspect=1.8)
# g.set(title="Revenue Distribution after Log Transformation", xlabel="log(Revenue)", ylabel="Count")
####IMG

trainX.corr()['revenue'].sort_values()

"""## ML Models"""

X = trainX[trainX.columns.difference(['budget', 'revenue', 'log_revenue'])]
y = trainX['log_revenue']

"""#### train-test split"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=10, shuffle=True)
y_train = y_train.values.reshape(-1,1)
y_test = y_test.values.reshape(-1,1)

"""Normalization"""

scaler = MinMaxScaler()
target_scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
y_train_norm = target_scaler.fit_transform(y_train)

X_test_norm = scaler.transform(X_test)
y_test_norm = target_scaler.transform(y_test)

from joblib import dump, load
import pickle
import os

"""### Linear Regression"""
lr_path = 'lr.joblib'
ridge_path = 'ridge.joblib'
lasso_path = 'lasso.joblib'
dt_path = 'dt.joblib'
rf_path = 'rf.joblib'
knn_path = 'knn.joblib'


if os.path.exists(lr_path):
  lr_model = load(lr_path)
else:
  lr_model = LinearRegression().fit(X_train_norm, y_train_norm)
  dump(lr_model, lr_path)
  

y_pred_norm = lr_model.predict(X_test_norm)
y_pred_norm = y_pred_norm.flatten()

print("Mean Absolute Error: ", mean_absolute_error(y_pred_norm, y_test_norm))
print("r2_score: ", r2_score(y_test_norm, y_pred_norm))

lr_model.score(X_test_norm, y_test_norm)

"""### Ridge Regression"""


if os.path.exists(ridge_path):
  ridge_model = load(ridge_path)
else:
  ridge_model = Ridge(alpha=0.5)
  ridge_model.fit(X_train_norm, y_train_norm)
  dump(ridge_model, ridge_path)
  

y_pred_norm = ridge_model.predict(X_test_norm)

print("Mean Absolute Errore: ", mean_absolute_error(y_pred_norm, y_test_norm))
print("r2_score: ", r2_score(y_test_norm, y_pred_norm))

ridge_model.score(X_test_norm, y_test_norm)

# plt.scatter(y_test_norm, y_pred_norm)
####IMG

"""### Lasso Regression"""

if os.path.exists(lasso_path):
  lasso_model = load(lasso_path)
else:
  lasso_model = Lasso(alpha=0.000001)
  lasso_model.fit(X_train_norm, y_train_norm)
  dump(lasso_model, lasso_path)

y_pred_norm = lasso_model.predict(X_test_norm)

print("Mean Absolute Error: ", mean_absolute_error(y_pred_norm, y_test_norm))
print("r2_score: ", r2_score(y_test_norm, y_pred_norm))

lasso_model.score(X_test_norm, y_test_norm)

# plt.scatter(y_test_norm, y_pred_norm)
####IMG

"""### Decision Tree"""

if os.path.exists(dt_path):
  dt_regressor = load(dt_path)
else:
  dt_regressor = DecisionTreeRegressor(max_depth=5, random_state=0)
  dt_regressor.fit(X_train_norm, y_train_norm)
  dump(dt_regressor, dt_path)

y_pred_norm = dt_regressor.predict(X_test_norm)

print("Mean Absolute Error: ", mean_absolute_error(y_pred_norm, y_test_norm))
print("r2_score: ", r2_score(y_test_norm, y_pred_norm))
print("RMSLE: ", mean_squared_log_error(y_test_norm, y_pred_norm))

dt_regressor.score(X_test_norm, y_test_norm)

"""### Random Forest"""


if os.path.exists(rf_path):
  rf_regressor = load(rf_path)
else:
  rf_regressor = RandomForestRegressor(max_depth=13, random_state=0)
  rf_regressor.fit(X_train_norm, y_train_norm)
  dump(rf_regressor, rf_path)

y_pred_norm = rf_regressor.predict(X_test_norm)

print("Mean Absolute Error: ", mean_absolute_error(y_pred_norm, y_test_norm))
print("r2_score: ", r2_score(y_test_norm, y_pred_norm))
print("RMSLE: ", mean_squared_log_error(y_test_norm, y_pred_norm))

rf_regressor.score(X_test_norm, y_test_norm)

# plt.scatter(y_test_norm, y_pred_norm)
####IMG

"""### KNN"""


if os.path.exists(knn_path):
  neigh_model = load(knn_path)
else:

  neigh_model = KNeighborsRegressor(n_neighbors=13, metric='manhattan')
  neigh_model.fit(X_train_norm, y_train_norm)
  dump(neigh_model, knn_path)

neigh_model = KNeighborsRegressor(n_neighbors=13, metric='manhattan')
neigh_model.fit(X_train_norm, y_train_norm)

y_pred_norm = neigh_model.predict(X_test_norm)
y_pred_norm[0:10]

print("Mean Absolute Error: ", mean_absolute_error(y_pred_norm, y_test_norm))
print("r2_score: ", r2_score(y_test_norm, y_pred_norm))
print("RMSLE: ", mean_squared_log_error(y_test_norm, y_pred_norm))

neigh_model.score(X_test_norm, y_test_norm)

"""### Neural Network"""

from keras.layers import Dense
from keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

nn_path = './nn.model'
if os.path.exists(nn_path):
  model = keras.models.load_model(nn_path)
else:
  model = Sequential()
  model.add(Dense(32, input_shape=(32,), activation='sigmoid'))
  model.add(Dense(1, activation='relu'))
  model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01))
  model.summary()

  history = model.fit(X_train_norm, y_train_norm, epochs=40, validation_data=(X_test_norm, y_test_norm))
  model.save(nn_path)

y_pred_norm = model.predict(X_train_norm)
print("Mean Absolute Error: ", mean_absolute_error(y_pred_norm, y_train_norm))
print("r2_score: ", r2_score(y_train_norm, y_pred_norm))
print("RMSLE: ", mean_squared_log_error(y_train_norm, y_pred_norm))

y_pred_norm = model.predict(X_test_norm)
print("Mean Absolute Error: ", mean_absolute_error(y_pred_norm, y_test_norm))
print("r2_score: ", r2_score(y_test_norm, y_pred_norm))
print("RMSLE: ", mean_squared_log_error(y_test_norm, y_pred_norm))

def get_scores():
  scores = []
  
  y_pred_norm = lr_model.predict(X_test_norm)
  y_pred_norm = y_pred_norm.flatten()
  scores.append(['Linear Regression', mean_absolute_error(y_pred_norm, y_test_norm), r2_score(y_test_norm, y_pred_norm), '-'])
  
  y_pred_norm = model.predict(X_test_norm)
  scores.append(['Neural Network', mean_absolute_error(y_pred_norm, y_test_norm), r2_score(y_test_norm, y_pred_norm),mean_absolute_error(y_pred_norm, y_test_norm)])
    
  y_pred_norm = ridge_model.predict(X_test_norm)
  scores.append(['Ridge Regressor', mean_absolute_error(y_pred_norm, y_test_norm), r2_score(y_test_norm, y_pred_norm), '-'])

  y_pred_norm = lasso_model.predict(X_test_norm)
  scores.append(['Lasso Regressor', mean_absolute_error(y_pred_norm, y_test_norm), r2_score(y_test_norm, y_pred_norm), '-'])
  
  y_pred_norm = dt_regressor.predict(X_test_norm)
  scores.append(['Decision Tree', mean_absolute_error(y_pred_norm, y_test_norm), r2_score(y_test_norm, y_pred_norm),mean_absolute_error(y_pred_norm, y_test_norm)])
  
  y_pred_norm = rf_regressor.predict(X_test_norm)
  scores.append(['Random Forest', mean_absolute_error(y_pred_norm, y_test_norm), r2_score(y_test_norm, y_pred_norm),mean_absolute_error(y_pred_norm, y_test_norm)])

  y_pred_norm = neigh_model.predict(X_test_norm)
  scores.append(['K-Nearest Neighbor', mean_absolute_error(y_pred_norm, y_test_norm), r2_score(y_test_norm, y_pred_norm),mean_absolute_error(y_pred_norm, y_test_norm)])
  
  return scores


from io import BytesIO
import base64

def get_fig():
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def get_plots(n):
  n = n/100
  print(type(n), n)
  matplotlib.use('TkAgg')
  
  train_plt = train.sample(frac=n)
  plots = []
  sns.catplot(x = 'has_collection', y = 'revenue', data=train_plt, height=7, aspect=0.8)
  plots.append(
    {
      'name': 'Impact of having a collection on movie revenue',
      'plot_url': get_fig()
    }
  )
  
  sns.set_theme(style="ticks")
  sns.catplot(x='has_homepage', y='revenue', data=train_plt, height=7, aspect=0.8)
  plots.append(
    {
      'name': 'Impact of having a home page on movie revenue',
      'plot_url': get_fig()
    }
  )
  
  sns.set(rc = {'figure.figsize':(15,8)})
  sns.set_style("whitegrid", {'axes.grid' : False})
  sns.scatterplot(x='budget', y='revenue', data=train_plt)
  plots.append(
    {
      'name': 'Impact of budget on movie revenue',
      'plot_url': get_fig()
    }
  )
    
  sns.scatterplot(x='popularity', y='revenue', data=train_plt[train_plt['popularity']<50])
  plots.append(
    {
      'name': 'Impact of popularity on movie revenue',
      'plot_url': get_fig()
    }
  )

  sns.scatterplot(x='runtime', y='revenue', data=train_plt)
  plots.append(
    {
      'name': 'Impact of runtime on movie revenue',
      'plot_url': get_fig()
    }
  )
  
  prod_companies_plt = pd.DataFrame(pd.read_csv('train.csv').sample(frac=n)['production_companies'].apply(lambda x: pd.Series(x)).stack().value_counts()).reset_index()
  prod_companies_plt.columns= ['companies', 'movies_count']
  sns.set_theme(style = "darkgrid")
  sns.set(rc={'figure.figsize':(18, 14)})
  sns.barplot(y="companies", x="movies_count", data=prod_companies_plt.head(50))
  plots.append(
    {
      'name': 'Movie count of production companies',
      'plot_url': get_fig()
    }
  )
  
  sns.set(rc = {'figure.figsize':(15,8)})
  sns.set_style("whitegrid", {'axes.grid' : False})
  sns.catplot(x='num_of_companies', y='revenue', data=train_plt, height=7, aspect=0.9)
  plots.append(
    {
      'name': 'Graph of number of companies and revenue',
      'plot_url': get_fig()
    }
  )
  
  prod_companies_plt = pd.DataFrame(pd.read_csv('train.csv').sample(frac=n)['production_countries'].apply(lambda x: pd.Series(x)).stack().value_counts()).reset_index()
  prod_companies_plt.columns= ['countries', 'movies_count']
  sns.set_theme(style = "darkgrid")
  sns.set(rc={'figure.figsize':(20, 8)})
  sns.barplot(y="countries", x="movies_count", data=prod_companies_plt.head(50))
  plots.append(
    {
      'name': 'Movie count of countries',
      'plot_url': get_fig()
    }
  )
  
  sns.catplot(x='usa_produced', y='revenue', data=train_plt, height=7, aspect=0.9)
  plots.append(
    {
      'name': 'Impact of USA production on movie revenue',
      'plot_url': get_fig()
    }
  )
  
  sns.catplot(x='is_released', y='revenue', data=train_plt, height=7, aspect=0.9)
  plots.append(
    {
      'name': 'Impact of being released on movie revenue',
      'plot_url': get_fig()
    }
  )
  
  sns.catplot(x='budget', y='release_year', data=train_plt, orient="h", height=14, aspect=0.7)
  plots.append(
    {
      'name': 'Impact of release year on movie revenue',
      'plot_url': get_fig()
    }
  )
  
  sns.catplot(y='revenue', x='release_month', data=train_plt, orient="v", height=8, aspect=1.3)
  plots.append(
    {
      'name': 'Impact of release month on movie revenue',
      'plot_url': get_fig()
    }
  )
  
  sns.catplot(x='num_of_spoken_languages', y='revenue', data=train_plt, height=8)
  plots.append(
    {
      'name': 'Impact of number of spoken languages on movie revenue',
      'plot_url': get_fig()
    }
  )
  
    
  train_plt['log_budget'] = np.log1p(train_plt['budget'])
  train_plt['log_revenue'] = np.log1p(train_plt['revenue'])
    
  g = sns.displot(data=train_plt, x="budget", height=8, aspect=1.8, binwidth=10000000)
  g.set(title="Budget Distribution before Log Transformation", xlabel="Budget", ylabel="Count")
  plots.append(
    {
      'name': 'Budget Distribution before Log Transformation',
      'plot_url': get_fig()
    }
  )
  
  g = sns.displot(data=train_plt, x="log_budget", height=8, aspect=1.8)
  g.set(title="Budget Distribution after Log Transformation", xlabel="log(Budget)", ylabel="Count")
  plots.append(
    {
      'name': 'Budget Distribution after Log Transformation',
      'plot_url': get_fig()
    }
  )
  
  g = sns.displot(data=train_plt, x="revenue", height=8, aspect=1.8, binwidth=10000000)
  g.set(title="Revenue Distribution before Log Transformation", xlabel="Revenue", ylabel="Count", xlim=(0,600000000))
  plots.append(
    {
      'name': 'Revenue Distribution before Log Transformation',
      'plot_url': get_fig()
    }
  )
  
  g = sns.displot(data=train_plt, x="log_revenue", height=8, aspect=1.8)
  g.set(title="Revenue Distribution after Log Transformation", xlabel="log(Revenue)", ylabel="Count")
  plots.append(
    {
      'name': 'Revenue Distribution after Log Transformation',
      'plot_url': get_fig()
    }
  )
  
  return plots
  
  
def get_predictions(id):
  train = pd.read_csv('train.csv')
  train = train[train['id'] == id]
  train.drop(['id', 'imdb_id', 'overview', 'poster_path', 'title', 'tagline', 'Keywords', 'belongs_to_collection', 'homepage', 'original_title', 'original_language', 'production_companies', 'production_countries', 'release_date', 'spoken_languages', 'status', 'cast', 'crew'], axis=1, inplace=True)
  train['genres'] = train['genres'].fillna('[]').map(eval).map(lambda x: [g['name'].lower() for g in x])
    
  mlb = MultiLabelBinarizer()
  genre_mlb = mlb.fit_transform(train['genres'])
  genre_labels = mlb.classes_
  genre_df = pd.DataFrame(genre_mlb, columns=genre_labels)

  train = train.join(genre_df)

  train.drop(['genres'], axis=1, inplace=True)
  train['log_budget'] = np.log1p(train['budget'])
  train['log_revenue'] = np.log1p(train['revenue'])
