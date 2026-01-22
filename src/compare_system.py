from baseline_base_prediction import get_baseline_pred
from regime_based_prediction import get_regime_df

df_regime=get_regime_df()
df_baseline=get_baseline_pred()

matches_baseline = (df_baseline['target'] == df_baseline['baseline_pred'])
accuracy_baseline=(matches_baseline.sum())/len(df_baseline)

print("baseline model accuracy : ",accuracy_baseline)
print("baseline model matches total : ",matches_baseline.sum())


matches_regime = (df_regime['target'] == df_regime['regime_pred'])
accuracy_regime=(matches_regime.sum())/len(df_regime)

print("regime model accuracy : ",accuracy_regime)
print("regime model matches total : ",matches_regime.sum())