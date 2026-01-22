from data_loader import load_data
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
import joblib
import os

df_train=load_data(r"C:\Users\Rehan Imtiyaj Mulla\OneDrive\Documents\PROJECT FILES\RegimeX\dataset\df_train.csv")
df_test=load_data(r"C:\Users\Rehan Imtiyaj Mulla\OneDrive\Documents\PROJECT FILES\RegimeX\dataset\df_test.csv")

x_train=df_train[["return","5_d_mean_return","21_d_mean_return","rsi14","macd_histogram",
      "10_d_volatility",
      "atr_14_d",
      "(SMA20 − SMA50)per_Price",
      "trend_strength_sma50"]]
y_train=df_train[["target"]]

x_test=df_test[["return","5_d_mean_return","21_d_mean_return","rsi14","macd_histogram",
      "10_d_volatility",
      "atr_14_d",
      "(SMA20 − SMA50)per_Price",
      "trend_strength_sma50"]]
y_test=df_test[["target"]]

baseline_model=XGBClassifier(n_estimators=100,max_depth=5,learning_rate=0.1,random_state=42)
baseline_model.fit(x_train,y_train)

y_pred=baseline_model.predict(x_test)

print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

folder_name = 'models'
file_name = 'baseline_model'
save_path = os.path.join(folder_name, file_name)
joblib.dump(baseline_model, save_path)
print(f"Model successfully saved to: {save_path}")