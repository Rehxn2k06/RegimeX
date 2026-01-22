from data_loader import load_data
import pandas as pd 
import joblib

base_model=joblib.load("models/baseline_model")
stress_model=joblib.load("models/stress_model")
calm_model=joblib.load("models/calm_model")


df_interpret=load_data(r"C:\Users\Rehan Imtiyaj Mulla\OneDrive\Documents\PROJECT FILES\RegimeX\dataset\df_test.csv")

calm_df=df_interpret.loc[df_interpret['p(stress)'] <= 0.3 , ["return","5_d_mean_return","21_d_mean_return","rsi14","macd_histogram",
      "10_d_volatility",
      "atr_14_d",
      "(SMA20 − SMA50)per_Price",
      "trend_strength_sma50","target","p(stress)"]]

stress_df = df_interpret.loc[df_interpret['p(stress)'] >= 0.7 , ["return","5_d_mean_return","21_d_mean_return","rsi14","macd_histogram",
      "10_d_volatility",
      "atr_14_d",
      "(SMA20 − SMA50)per_Price",
      "trend_strength_sma50","target","p(stress)"]]

transition_df=df_interpret.loc[(df_interpret['p(stress)'] > 0.3) & (df_interpret['p(stress)'] < 0.7) , ["return","5_d_mean_return","21_d_mean_return","rsi14","macd_histogram","10_d_volatility","atr_14_d","(SMA20 − SMA50)per_Price","trend_strength_sma50","target","p(stress)"]]


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


y_stress_regime=stress_model.predict(x_stress)
y_calm_regime=calm_model.predict(x_calm)
y_transition_regime=base_model.predict(x_transition)


y_baseline_stress=base_model.predict(x_stress)
y_baseline_calm=base_model.predict(x_calm)
y_baseline_transition=base_model.predict(x_transition)

calm_df["regime_preds"]=y_calm_regime
calm_df["baseline_pred"]=y_baseline_calm


stress_df["regime_preds"]=y_stress_regime
stress_df["baseline_pred"]=y_baseline_stress

transition_df["regime_preds"]=y_transition_regime
transition_df["baseline_pred"]=y_baseline_transition

'''
print(calm_df.head())
print(stress_df.head())
print(transition_df.head())'''

matches_regime_calm = (calm_df['target'] == calm_df['regime_preds'])
accuracy_regime_calm=(matches_regime_calm.sum())/len(calm_df)
matches_regime_baseline = (calm_df['target'] == calm_df['baseline_pred'])
accuracy_regime_baseline=(matches_regime_baseline.sum())/len(calm_df)

print("the accuracy for baseline model in calm regime is - " ,accuracy_regime_baseline )
print("the accuracy for regime model in calm regime is - " ,accuracy_regime_calm )
print("\n")


matches_regime_stress = (stress_df['target'] == stress_df['regime_preds'])
accuracy_regime_stress=(matches_regime_stress.sum())/len(stress_df)
matches_baseline_stress = (stress_df['target'] == stress_df['baseline_pred'])
accuracy_baseline_stress=(matches_baseline_stress.sum())/len(stress_df)

print("the accuracy for baseline model in stress regime is - " ,accuracy_regime_stress )
print("the accuracy for regime model in stress regime is - " ,accuracy_baseline_stress )
print("\n")


matches_regime_transition = (transition_df['target'] == transition_df['regime_preds'])
accuracy_regime_transition=(matches_regime_transition.sum())/len(transition_df)
matches_baseline_transition = (transition_df['target'] == transition_df['baseline_pred'])
accuracy_baseline_transition=(matches_baseline_transition.sum())/len(transition_df)

print("the accuracy for baseline model in transition regime is - " ,accuracy_regime_transition )
print("the accuracy for regime model in transition regime is - " ,accuracy_baseline_transition )
print("\n")

