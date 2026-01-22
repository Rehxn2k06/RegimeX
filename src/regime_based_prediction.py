from data_loader import load_data
import pandas as pd 
import joblib
import os 
import numpy as np

base_model=joblib.load("models/baseline_model")
stress_model=joblib.load("models/stress_model")
calm_model=joblib.load("models/calm_model")


df_pred_regime=load_data(r"C:\Users\Rehan Imtiyaj Mulla\OneDrive\Documents\PROJECT FILES\RegimeX\dataset\df_test.csv")

calm_df=df_pred_regime.loc[df_pred_regime['p(stress)'] <= 0.3 , ["return","5_d_mean_return","21_d_mean_return","rsi14","macd_histogram",
      "10_d_volatility",
      "atr_14_d",
      "(SMA20 − SMA50)per_Price",
      "trend_strength_sma50","target","p(stress)"]]

stress_df = df_pred_regime.loc[df_pred_regime['p(stress)'] >= 0.7 , ["return","5_d_mean_return","21_d_mean_return","rsi14","macd_histogram",
      "10_d_volatility",
      "atr_14_d",
      "(SMA20 − SMA50)per_Price",
      "trend_strength_sma50","target","p(stress)"]]

transition_df=df_pred_regime.loc[(df_pred_regime['p(stress)'] > 0.3) & (df_pred_regime['p(stress)'] < 0.7) , ["return","5_d_mean_return","21_d_mean_return","rsi14","macd_histogram","10_d_volatility","atr_14_d","(SMA20 − SMA50)per_Price","trend_strength_sma50","target","p(stress)"]]

x_stress=stress_df[["return","5_d_mean_return","21_d_mean_return","rsi14","macd_histogram",
      "10_d_volatility",
      "atr_14_d",
      "(SMA20 − SMA50)per_Price",
      "trend_strength_sma50"]]

x_calm=calm_df[["return","5_d_mean_return","21_d_mean_return","rsi14","macd_histogram",
      "10_d_volatility",
      "atr_14_d",
      "(SMA20 − SMA50)per_Price",
      "trend_strength_sma50"]]

x_transition=transition_df[["return","5_d_mean_return","21_d_mean_return","rsi14","macd_histogram",
      "10_d_volatility",
      "atr_14_d",
      "(SMA20 − SMA50)per_Price",
      "trend_strength_sma50"]]


y_stress=stress_model.predict(x_stress)
y_calm=calm_model.predict(x_calm)
y_transition=base_model.predict(x_transition)

df_pred_regime["regime_pred"]=np.nan
df_pred_regime.loc[calm_df.index,"regime_pred"]=y_calm
df_pred_regime.loc[stress_df.index,"regime_pred"]=y_stress
df_pred_regime.loc[transition_df.index,"regime_pred"]=y_transition

if __name__ == "__main__":
     print(df_pred_regime.head())
     print(df_pred_regime["regime_pred"].isnull().sum())

def get_regime_df():
    return df_pred_regime