# %% [code] {"execution":{"iopub.status.busy":"2024-01-10T08:55:45.315918Z","iopub.execute_input":"2024-01-10T08:55:45.316741Z","iopub.status.idle":"2024-01-10T08:55:47.230037Z","shell.execute_reply.started":"2024-01-10T08:55:45.316701Z","shell.execute_reply":"2024-01-10T08:55:47.228375Z"}}
import sys
import polars as pl
import polars.selectors as cs
import holidays
from sklearn.model_selection import train_test_split
import os
class Dataset():
    def __init__(self):
        self.IS_COLAB = ~os.path.isfile("/home/davidgauthier/Codes/Hackatons_2024/Enefit/data/train.csv")
        self.IS_KAGGLE = "kaggle_secrets" in sys.modules
        if self.IS_KAGGLE:
            self.path = '/kaggle/input/predict-energy-behavior-of-prosumers'
        else:
            self.path = "/home/davidgauthier/Codes/Hackatons_2024/Enefit/data"

    def load(self):
        # Data
        self.info      = pl.read_csv(self.path+'/train.csv', try_parse_dates=True).drop_nulls('target')
        self.client    = pl.read_csv(self.path+'/client.csv', try_parse_dates=True)
        self.e_price   = pl.read_csv(self.path+'/electricity_prices.csv', try_parse_dates=True)
        self.g_price   = pl.read_csv(self.path+'/gas_prices.csv', try_parse_dates=True)
        self.f_weather = pl.read_csv(self.path+'/forecast_weather.csv', try_parse_dates=True)
        self.h_weather = pl.read_csv(self.path+'/historical_weather.csv', try_parse_dates=True)
        
        self.ws_county = pl.read_csv(self.path+'/weather_station_to_county_mapping.csv', try_parse_dates=True)
        self.county_pos = self.ws_county.drop_nulls().sort('county_name')\
                .group_by(pl.col('county')).agg(
                        pl.col('longitude').map_elements(lambda x: [min(x),max(x)]),
                        pl.col('latitude').map_elements(lambda x: [min(x),max(x)]))

        # Splits
        blocks                       = self.info['data_block_id'].unique().to_numpy()
        self.train_id, self.test_id  = train_test_split(blocks, shuffle=False, test_size=0.3)

    def time_features(self):
        holidays_est = holidays.Estonia()
        is_holidays  = pl.col('datetime').dt.strftime('%Y-%m-%d').is_in([x.strftime('%Y-%m-%d') for x in holidays_est.keys()]).cast(int)
        train_info = train_info.with_columns(is_holidays=is_holidays, weekdays=pl.col('datetime').dt.weekday, months=pl.col('datetime').dt.month,)

    def add_features(self):
        pass

    def pl_train_test_split(self,df):    
        return df.filter(pl.col('data_block_id').is_in(self.train_id)),\
                df.filter(pl.col('data_block_id').is_in(self.valid_id)),\
                 df.filter(pl.col('data_block_id').is_in(self.test_id))
    
    def meta_(self):
        self.info.select(~cs.matches('.*_id')).describe()


    def get_county_from_map(self, lat, lon, county_pos) -> pl.Expr:
        return pl.when(pl.col('latitude').is_between(*lat) & pl.col('longitude').is_between(*lon)).then(county_pos).otherwise(-1).alias('county_in_'+str(county_pos))

    def loc_county(self, weather):
        '''Find Estonian county based on coordinates'''
        return weather.with_columns(weather.with_columns([self.get_county_from_map(lat, lon, county) for county, lon, lat in self.county_pos.iter_rows()]).select(cs.matches('county_in_\d*')).max_horizontal().alias('county').cast(pl.Int64)).filter(~(pl.col('county')==-1))

    def fill_weather(self, weather): 
        return pl.concat((weather,weather.filter(pl.col('county')==10).with_columns(county=pl.lit(1,dtype=pl.Int64)),weather.filter(pl.col('county')==11).with_columns(county=pl.lit(8,dtype=pl.Int64))))

    def merge_set(self):

        # Capacity production (prices)
        production = self.info

        # Add County and fill missing with closest region
        h_weather = self.loc_county(self.h_weather)
        h_weather = self.fill_weather(h_weather)

        f_weather = self.loc_county(self.f_weather)
        f_weather = self.fill_weather(f_weather)

        # Merge production data and weather data
        production = self.production_meteo_(f_weather,production)

        # FB Fill for missing weather forecasts
        production = self.fill_missings(production)

        return production

    def production_meteo_(self, f_weather,production):
        # Subset Weather
        f_weather = f_weather.filter(pl.col('hours_ahead').is_between(0,25)).sort(['county','forecast_datetime'])
        # Average across counties !MEAN!
        f_weather = f_weather.group_by(['forecast_datetime','county']).mean()
        # Name compatibilty
        f_weather = f_weather.rename({'forecast_datetime':'datetime'})
        # Merge production + weather forecasts
        return production.join(f_weather, how='left', on=['county','datetime'])

    def fill_missings(self, production_meteo):
        # Take last meteo forecast
        production_meteo = production_meteo.group_by('county').map_groups(lambda x: x.with_columns(pl.all().forward_fill()).with_columns(pl.all().backward_fill()))
        return production_meteo
    
    def update_data(self,df_new_client,df_new_gas_prices,df_new_electricity_prices,df_new_forecast_weather,df_new_historical_weather,df_new_target):

        df_new_client = pl.from_pandas(
            df_new_client[self.client.drop(columns=['data_block_id']).columns], schema_overrides=self.client.schema
        )
        df_new_gas_prices = pl.from_pandas(
            df_new_gas_prices[self.g_price.drop(columns=['data_block_id']).columns], schema_overrides=self.g_price.schema
        )
        df_new_electricity_prices = pl.from_pandas(
            df_new_electricity_prices[self.e_price.drop(columns=['data_block_id']).columns], schema_overrides=self.e_price.schema
        )
        df_new_forecast_weather = pl.from_pandas(
            df_new_forecast_weather[self.f_weather.drop(columns=['data_block_id']).columns], schema_overrides=self.f_weather.schema
        )
        df_new_historical_weather = pl.from_pandas(
            df_new_historical_weather[self.h_weather.drop(columns=['data_block_id']).columns], schema_overrides=self.h_weather.schema
        )
        df_new_target = pl.from_pandas(
            df_new_target[self.info.drop(columns=['data_block_id']).columns], schema_overrides=self.info.schema
        )

        self.client    = pl.concat((self.client.drop(columns=['data_block_id']),df_new_client))
        self.g_price   = pl.concat((self.g_price.drop(columns=['data_block_id']),df_new_gas_prices))
        self.e_price   = pl.concat((self.e_price.drop(columns=['data_block_id']),df_new_electricity_prices))
        self.f_weather = pl.concat((self.f_weather.drop(columns=['data_block_id']),df_new_forecast_weather))
        self.h_weather = pl.concat((self.h_weather.drop(columns=['data_block_id']),df_new_historical_weather))
        self.info      = pl.concat((self.info.drop(columns=['data_block_id']),df_new_target))




#     def cons