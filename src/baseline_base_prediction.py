from data_loader import load_data
import pandas as pd 
import joblib
import os 
import numpy as np

base_model=joblib.load("models/baseline_model")

df_baseline=load_data(r"C:\Users\Rehan Imtiyaj Mulla\OneDrive\Documents\PROJECT FILES\RegimeX\dataset\df_test.csv")

x_base=df_baseline[["return","5_d_mean_return","21_d_mean_return","rsi14","macd_histogram",
      "10_d_volatility",
      "atr_14_d",
      "(SMA20 − SMA50)per_Price",
      "trend_strength_sma50"]]

y__base_pred=base_model.predict(x_base)

df_baseline["baseline_pred"]=y__base_pred

if __name__ == "__main__":
    print(df_baseline.head())
    print(df_baseline["baseline_pred"].isna().sum())

def get_baseline_pred():
    return df_baseline