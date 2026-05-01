import pandas as pd
import numpy as np
from regime_based_prediction import get_regime_df
from baseline_base_prediction import get_baseline_pred
import matplotlib.pyplot as plt

df_regime=get_regime_df()
df_baseline=get_baseline_pred()

df_regime["position"]=np.where(df_regime["regime_pred"]==1,1,-1)
df_regime["strategy_return"]=df_regime["position"]*df_regime["return"]
df_regime["cummulative_strategy_returns"]=df_regime["strategy_return"].cumsum()



df_baseline["position"]=np.where(df_baseline["baseline_pred"]==1,1,-1)
df_baseline["strategy_return"]=df_baseline["position"]*df_baseline["return"]
df_baseline["cummulative_strategy_returns"]=df_baseline["strategy_return"].cumsum()


plt.plot(df_baseline["Date"],df_baseline["cummulative_strategy_returns"],label="Baseline system",color="red")
plt.plot(df_baseline["Date"],df_regime["cummulative_strategy_returns"],label="Regime-aware system",color="blue")
plt.title("Cumulative Returns — Baseline vs Regime-Aware (Test Set)")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.tight_layout() 
plt.show()