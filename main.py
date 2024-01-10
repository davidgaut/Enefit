
#%%
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

import pandas as pd
import polars as pl
import polars.selectors as cs

import numpy as np

import random
import os

from functions import Dataset

plt.style.use('seaborn')
rcParams["figure.figsize"] = [15,10]

def seed_everything(seed,tensorflow_init=True,pytorch_init=True):
    """
    Seeds basic parameters for reproducibility of results
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if tensorflow_init is True:
        tf.random.set_seed(seed)
    if pytorch_init is True:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Seed
seed_everything(42,tensorflow_init=False,pytorch_init=False)


# %%

# Dataset
dataset = Dataset()
dataset.load()

train_info   = dataset.train_info
train_client = dataset.train_client
train_e_prices  = dataset.train_e_price
train_g_prices  = dataset.train_g_price
train_h_weather = dataset.train_h_weather

#
dd = train_info.sort('datetime').filter(pl.col('is_consumption')==1,pl.col('is_business')==0)\
    .rolling(index_column='datetime',period='1d',by='prediction_unit_id').agg(pl.col('target').mean()/pl.col('target').std())\
    .with_columns(pl.col('target'))

#
sns.lineplot(data=dd,x='datetime',y='target',errorbar=None,hue='prediction_unit_id')

# By county (consumption for 13,22,30 is less correlated)
_,ax = plt.subplots(figsize=(15,15))
sns.heatmap(dd.pivot(columns='prediction_unit_id',values='target',index='datetime').fill_null(0).select(cs.float()).corr(), ax=ax)

#
dd = train_info.sort('datetime').filter(pl.col('is_consumption')==0,pl.col('is_business')==0)\
    .rolling(index_column='datetime',period='1d',by='prediction_unit_id').agg(pl.col('target').mean()/pl.col('target').std())\
    .with_columns(pl.col('target'))

#
sns.lineplot(data=dd,x='datetime',y='target',errorbar=None,hue='prediction_unit_id')

# By county (consumption for 13,22,30 is less correlated)
_,ax = plt.subplots(figsize=(15,15))
sns.heatmap(dd.pivot(columns='prediction_unit_id',values='target',index='datetime').fill_null(0).select(cs.float()).corr(), ax=ax)

#
f,axes = plt.subplots(4,1,sharex=True)
sns.lineplot(data=train_e_prices,x='forecast_date',y='euros_per_mwh')
sns.lineplot(data=train_e_prices.sort('forecast_date').rolling(index_column='forecast_date',period='1d').agg(pl.col('euros_per_mwh').mean()),x='forecast_date',y='euros_per_mwh')
sns.lineplot(data=train_g_prices,x='forecast_date',y='lowest_price_per_mwh')
sns.lineplot(data=train_g_prices,x='forecast_date',y='highest_price_per_mwh')

dd = train_info.sort('datetime').filter(pl.col('is_consumption')==1,pl.col('is_business')==0)\
    # .rolling(index_column='datetime',period='1d',by='prediction_unit_id').agg(pl.col('target').mean()/pl.col('target').std())\
    # .with_columns(pl.col('target'))

sns.lineplot(data=dd,x='datetime',y='target',errorbar=None,hue='prediction_unit_id',ax=axes[0])

dd = train_info.sort('datetime').filter(pl.col('is_consumption')==0,pl.col('is_business')==0)\
    # .rolling(index_column='datetime',period='1d',by='prediction_unit_id').agg(pl.col('target').mean()/pl.col('target').std())\
    # .with_columns(pl.col('target'))

sns.lineplot(data=dd,x='datetime',y='target',errorbar=None,hue='prediction_unit_id',ax=axes[1])

dd = train_h_weather.group_by(['data_block_id','datetime']).agg(pl.col('direct_solar_radiation').mean())
sns.lineplot(data=dd,x='datetime',y='direct_solar_radiation',errorbar=None,ax=axes[2])

# dd = train_h_weather.group_by(['data_block_id','datetime']).agg(pl.col('temperature').mean())
# sns.lineplot(data=dd,x='datetime',y='temperature',errorbar=None,)




#%%

# %%
# Number of periods before production starts
segdate  = ['prediction_unit_id']
train_info.filter(pl.col('is_consumption')==0).with_columns(pl.lit(1)).select(pl.col('prediction_unit_id'), pl.col('datetime'), pl.col('target').cum_sum().over('prediction_unit_id'), pl.col('literal').cum_sum().over("prediction_unit_id")).filter(pl.col('target')==0).group_by("prediction_unit_id").agg(pl.col('literal').last()).filter(pl.col('literal').is_between(8,1000))

# %%
# Mean consumption / production across sectors
train_info.select(pl.col('county'),pl.col('target')).group_by('county').mean()
# %%
train_info.select(pl.col('product_type'),pl.col('target')).group_by('product_type').mean()
# %%
train_info.select(pl.col('is_business'),pl.col('target')).group_by('is_business').mean()

# %%
# Target in time
county_show = 10
data = train_info.filter(pl.col('county')==county_show)
g = sns.FacetGrid(data=data, row='product_type', col='is_business', aspect=2.5)
g.map(sns.lineplot,data=data, x='datetime', y='target', errorbar=None, hue='is_consumption')

# Target across counties
data = train_info.filter(pl.col('is_consumption')==1)
g = sns.FacetGrid(data=data, row='product_type', col='is_business', aspect=2.5)
g.map(sns.lineplot,data=data, x='datetime', y='target', errorbar=None, hue='county')

# %%
# Mean and Count
segments = ['product_type','is_business','is_consumption']
for cc in segments:
    mc = train_info.group_by(cc).agg(pl.col('target').count().name.suffix('_count'),pl.col('target').mean().name.suffix('_mean'))
    print(mc)

# %%
# Trends and Cycles
tmp = train_info.filter(pl.col('prediction_unit_id')==0,pl.col('is_consumption')==1)
_,axes = plt.subplots(2,1,figsize=(25,10),sharex=True)
ax = axes[0]
sns.lineplot(data=tmp,x='datetime',y='target',ax=ax)
sns.lineplot(data=tmp.sort('datetime').rolling(index_column='datetime',period='1d').agg(pl.col('target').mean()),x='datetime',y='target',ax=ax)
sns.lineplot(data=tmp.sort('datetime').rolling(index_column='datetime',period='7d').agg(pl.col('target').mean()),x='datetime',y='target',ax=ax)

tmp = train_info.filter(pl.col('prediction_unit_id')==0,pl.col('is_consumption')==0)
ax = axes[1]
sns.lineplot(data=tmp,x='datetime',y='target',ax=ax)
sns.lineplot(data=tmp.sort('datetime').rolling(index_column='datetime',period='1d').agg(pl.col('target').mean()),x='datetime',y='target',ax=ax)
sns.lineplot(data=tmp.sort('datetime').rolling(index_column='datetime',period='7d').agg(pl.col('target').mean()),x='datetime',y='target',ax=ax)

# Consumption and production during the day
conso_hours = train_info\
    .with_columns(hours=pl.col('datetime').dt.hour())\
    .group_by(['is_business','is_consumption','hours'])\
    .agg(pl.col('target').mean())

sns.catplot(
    data=conso_hours, x="hours", y="target", 
    row="is_consumption", hue='is_business',
    kind="bar", height=3, aspect=2.5,
)

# production is null between 20 and 6

# Across counties
conso_hours = train_info\
    .with_columns(hours=pl.col('datetime').dt.hour())\
    .group_by(['county','is_business','is_consumption','hours'])\
    .agg(pl.col('target').mean())

sns.catplot(
    data=conso_hours, x="hours", y="target", 
    row='is_business',col="is_consumption", hue='county',
    kind="bar", height=3, aspect=2.5,
)


sns.lineplot(data=train_info.filter(pl.col('prediction_unit_id')==1, pl.col('is_consumption')==0),x='datetime',y='target')

sns.lineplot(data=train_info.filter(pl.col('prediction_unit_id')==21, pl.col('is_consumption')==0),x='datetime',y='target')

# %%
# Some counties predict others

# %%
# Missings
train_info.with_columns(pl.all().is_null()).sum()
#
sns.scatterplot(data=train_info.with_columns(is_null=pl.col('target').is_null().cast(pl.Int16)*1), x='datetime',y='is_null')


# %%
# Correlations across dims

# Prod / Conso
pivot_ = train_info.pivot(columns=['is_consumption'],values=['target'],index=['county','product_type','is_business','datetime'])[['0','1']].drop_nulls()

pivot_.corr()
pivot_.filter(pl.col('0')!=0).corr()

# Business / Households
pivot_ = train_info.pivot(columns=['is_business'],values=['target'],index=['county','product_type','is_consumption','datetime']).drop_nulls()

pivot_.corr()
pivot_.group_by(pl.col('is_consumption')).agg(pl.corr('0','1'))

pivot_.group_by(pl.col('is_consumption')).agg(pl.corr('0','1'))

pivot_.group_by(pl.col('county')).agg(pl.corr('0','1'))

#
pivot_ = train_info.pivot(columns=['is_business','is_consumption'],values=['target'],index=['county','product_type','datetime'])#[['0','1']].drop_nulls()

pivot_.corr()
pivot_.filter(pl.col('0')!=0).corr()


#
train_info.pivot(columns=['county','is_consumption'],values='target',index=['datetime','prediction_unit_id',])

train_info.group_by('county').agg(pl.col('target')).explode(columns=['target'])

# train_info.to_pandas().groupby('is_business')[['target']].corr()

# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as mae

#
# Treatments: Trends

def make_data(df, Train=True):
    # Add dates
    # df = df.with_columns(day=pl.col('datetime').dt.day(),month=pl.col('datetime').dt.weekday())
    if Train:
        # Drop Nan target
        df = df.filter(~pl.col('target').is_null())

    # Dummy
    df = df.to_dummies(~cs.matches('.*_id$') & cs.integer(),drop_first=True)
    if Train:
        # Target
        y  = df['target']
        df = df.drop(columns='target')

    # Drop
    df = df.drop(columns=['datetime','target',cs.matches('.*_id$')])

    if Train:
        return df, y 

    return df
#
df_train, y_train = make_data(train_info)
df_valid, y_valid = make_data(valid_info)

#
lr = LinearRegression()
lr.fit(df_train, y_train)

train_pred = lr.predict(df_train) 
train_pred = np.array(list(map(lambda x: np.max((0.0,x)),train_pred)))

mae(y_train,train_pred)

valid_pred = lr.predict(df_valid)
valid_pred = np.array(list(map(lambda x: np.max((0.0,x)),valid_pred)))
mae(y_valid,valid_pred)

mae(y_valid,y_valid*0+0*y_train.mean())


# %%
import enefit
env       = enefit.make_env()
iter_test = env.iter_test()

counter = 0
for (test, revealed_targets, client, historical_weather,
        forecast_weather, electricity_prices, gas_prices, sample_prediction) in iter_test:

    #
    test = test.rename(columns={'prediction_datetime':'datetime'})
    cols = ['county', 'is_business', 'product_type', 'is_consumption', 'datetime']
    test = pl.DataFrame(test[cols],schema=train_info[cols].schema)

    # Data
    df_test = make_data(test,Train=False)

    # if counter == 0:
    #     print(test.head(3))
    #     print(revealed_targets.head(3))
    #     print(client.head(3))
    #     print(historical_weather.head(3))
    #     print(forecast_weather.head(3))
    #     print(electricity_prices.head(3))
    #     print(gas_prices.head(3))
    #     print(sample_prediction.head(3))
    
    # Pred
    sample_prediction['target'] = lr.predict(df_test)
    valid_pred = np.array(list(map(lambda x: np.max((0.0,x)),valid_pred)))

    env.predict(sample_prediction)
    counter += 1
