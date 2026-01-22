from data_loader import load_data
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
import joblib
import os

df_train=load_data(r"C:\Users\Rehan Imtiyaj Mulla\OneDrive\Documents\PROJECT FILES\RegimeX\dataset\df_train.csv")
df_test=load_data(r"C:\Users\Rehan Imtiyaj Mulla\OneDrive\Documents\PROJECT FILES\RegimeX\dataset\df_test.csv")
calm_df_train = df_train.loc[df_train['p(stress)'] <= 0.3 , ["return","5_d_mean_return","21_d_mean_return","rsi14","macd_histogram",
      "10_d_volatility",
      "atr_14_d",
      "(SMA20 − SMA50)per_Price",
      "trend_strength_sma50","target","p(stress)"]]

x_calm_train=calm_df_train[["return","5_d_mean_return","21_d_mean_return","rsi14","macd_histogram",
      "10_d_volatility",
      "atr_14_d",
      "(SMA20 − SMA50)per_Price",
      "trend_strength_sma50"]]

y_calm_train=calm_df_train[["target"]]

calm_df_test = df_test.loc[df_test['p(stress)'] <= 0.3 , ["return","5_d_mean_return","21_d_mean_return","rsi14","macd_histogram",
      "10_d_volatility",
      "atr_14_d",
      "(SMA20 − SMA50)per_Price",
      "trend_strength_sma50","target","p(stress)"]]

x_calm_test=calm_df_test[["return","5_d_mean_return","21_d_mean_return","rsi14","macd_histogram",
      "10_d_volatility",
      "atr_14_d",
      "(SMA20 − SMA50)per_Price",
      "trend_strength_sma50"]]

y_calm_test=calm_df_test[["target"]]



calm_model=XGBClassifier(n_estimators=100,max_depth=5,learning_rate=0.1,random_state=42)
calm_model.fit(x_calm_train,y_calm_train)

y_pred_c=calm_model.predict(x_calm_test)

print(accuracy_score(y_calm_test,y_pred_c))
print(classification_report(y_calm_test,y_pred_c))



folder_name = 'models'
file_name = 'calm_model'
save_path = os.path.join(folder_name, file_name)
joblib.dump(calm_model, save_path)
print(f"Model successfully saved to: {save_path}")