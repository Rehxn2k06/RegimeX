from data_loader import load_data
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
import joblib
import os

df_train=load_data(r"C:\Users\Rehan Imtiyaj Mulla\OneDrive\Documents\PROJECT FILES\RegimeX\dataset\df_train.csv")
df_test=load_data(r"C:\Users\Rehan Imtiyaj Mulla\OneDrive\Documents\PROJECT FILES\RegimeX\dataset\df_test.csv")
stress_df_train = df_train.loc[df_train['p(stress)'] >= 0.7 , ["return","5_d_mean_return","21_d_mean_return","rsi14","macd_histogram",
      "10_d_volatility",
      "atr_14_d",
      "(SMA20 − SMA50)per_Price",
      "trend_strength_sma50","target","p(stress)"]]

x_stress_train=stress_df_train[["return","5_d_mean_return","21_d_mean_return","rsi14","macd_histogram",
      "10_d_volatility",
      "atr_14_d",
      "(SMA20 − SMA50)per_Price",
      "trend_strength_sma50"]]

y_stress_train=stress_df_train[["target"]]

stress_df_test = df_test.loc[df_test['p(stress)'] >= 0.7 , ["return","5_d_mean_return","21_d_mean_return","rsi14","macd_histogram",
      "10_d_volatility",
      "atr_14_d",
      "(SMA20 − SMA50)per_Price",
      "trend_strength_sma50","target","p(stress)"]]

x_stress_test=stress_df_test[["return","5_d_mean_return","21_d_mean_return","rsi14","macd_histogram",
      "10_d_volatility",
      "atr_14_d",
      "(SMA20 − SMA50)per_Price",
      "trend_strength_sma50"]]

y_stress_test=stress_df_test[["target"]]



stress_model=XGBClassifier(n_estimators=100,max_depth=5,learning_rate=0.1,random_state=42)
stress_model.fit(x_stress_train,y_stress_train)

y_pred_c=stress_model.predict(x_stress_test)

print(accuracy_score(y_stress_test,y_pred_c))
print(classification_report(y_stress_test,y_pred_c))



folder_name = 'models'
file_name = 'stress_model'
save_path = os.path.join(folder_name, file_name)
joblib.dump(stress_model, save_path)
print(f"Model successfully saved to: {save_path}")