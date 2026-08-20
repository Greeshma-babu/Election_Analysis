import pathlib
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score, classification_report

# ---------------------------------------------------------
# Paths resolved relative to THIS script's location - works
# no matter what folder you run python from.
# ---------------------------------------------------------
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent   # up from machine_learning/ to Election_Analysis/
DATA_DIR = BASE_DIR / 'dataset' / 'training'
MODEL_DIR = BASE_DIR / 'models'

# ---------------------------------------------------------
# STEP 1: Load the three yearly election files
# ---------------------------------------------------------
df_2011 = pd.read_csv(DATA_DIR / 'election_2011.csv')
df_2016 = pd.read_csv(DATA_DIR / 'election_2016.csv')
df_2021 = pd.read_csv(DATA_DIR / 'election_2021.csv')

# ---------------------------------------------------------
# STEP 2: Rename columns per year so merging doesn't overwrite them
# ---------------------------------------------------------
df_2011 = df_2011.rename(columns={
    'result': 'result_2011', 'voter_turnout_pct': 'turnout_2011',
    'margin_of_victory_pct': 'margin_2011', 'swing_factor_pct': 'swing_2011'
}).drop(columns=['election_year'])

df_2016 = df_2016.rename(columns={
    'result': 'result_2016', 'voter_turnout_pct': 'turnout_2016',
    'margin_of_victory_pct': 'margin_2016', 'swing_factor_pct': 'swing_2016'
}).drop(columns=['election_year'])

df_2021 = df_2021.rename(columns={
    'result': 'result_2021', 'voter_turnout_pct': 'turnout_2021',
    'margin_of_victory_pct': 'margin_2021', 'swing_factor_pct': 'swing_2021'
}).drop(columns=['election_year'])

# ---------------------------------------------------------
# STEP 3: Merge into one row per constituency, all 3 years side by side
# ---------------------------------------------------------
merged = df_2011.merge(
    df_2016.drop(columns=['state', 'demographic']), on='constituency_id'
).merge(
    df_2021.drop(columns=['state', 'demographic']), on='constituency_id'
)

print("Merged shape:", merged.shape)

# ---------------------------------------------------------
# STEP 4: Feature engineering
# ---------------------------------------------------------
merged['turnout_change_16'] = merged['turnout_2016'] - merged['turnout_2011']
merged['turnout_change_21'] = merged['turnout_2021'] - merged['turnout_2016']
merged['margin_change_16'] = merged['margin_2016'] - merged['margin_2011']
merged['margin_change_21'] = merged['margin_2021'] - merged['margin_2016']
merged['seat_flip_16'] = (merged['result_2016'] != merged['result_2011']).astype(int)
merged['seat_flip_21'] = (merged['result_2021'] != merged['result_2016']).astype(int)
merged['retained_2021'] = (merged['margin_2021'] > 5).astype(int)

merged['demographic_encoded'] = merged['demographic'].map({'Urban': 0, 'Semi-Urban': 1, 'Rural': 2})

state_encoder = LabelEncoder()
merged['state_encoded'] = state_encoder.fit_transform(merged['state'])

party_encoder = LabelEncoder()
merged['result_2016_encoded'] = party_encoder.fit_transform(merged['result_2016'])

# ---------------------------------------------------------
# STEP 5: Build X (features) and the three y targets
# ---------------------------------------------------------
feature_cols = ['state_encoded', 'demographic_encoded', 'turnout_2016', 'turnout_2021',
                 'turnout_change_21', 'margin_2016', 'swing_2016', 'result_2016_encoded']

X = merged[feature_cols]
y_margin = merged['margin_2021']
y_retained = merged['retained_2021']
y_party = merged['result_2021']

# ---------------------------------------------------------
# STEP 6: One single split, shared across all three targets
# ---------------------------------------------------------
X_train, X_test, y_margin_train, y_margin_test, \
    y_retained_train, y_retained_test, \
    y_party_train, y_party_test = train_test_split(
        X, y_margin, y_retained, y_party, test_size=0.2, random_state=42
    )

# Save constituency_id aligned to the same split
id_train = merged.loc[X_train.index, 'constituency_id']
id_test = merged.loc[X_test.index, 'constituency_id']
id_test.to_csv(DATA_DIR / 'id_test.csv', index=False)

# Save the split to disk
X_train.to_csv(DATA_DIR / 'X_train.csv', index=False)
X_test.to_csv(DATA_DIR / 'X_test.csv', index=False)
y_margin_test.to_csv(DATA_DIR / 'y_test_margin.csv', index=False)
y_retained_test.to_csv(DATA_DIR / 'y_test_retained.csv', index=False)
y_party_test.to_csv(DATA_DIR / 'y_test_party.csv', index=False)

print("Saved id_test.csv, X_test.csv, and target files to", DATA_DIR)

# ---------------------------------------------------------
# STEP 7: Train models
# ---------------------------------------------------------
rf_margin = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
rf_margin.fit(X_train, y_margin_train)
print("Margin R2:", r2_score(y_margin_test, rf_margin.predict(X_test)))

rf_retained = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
rf_retained.fit(X_train, y_retained_train)
print("Retained accuracy:", accuracy_score(y_retained_test, rf_retained.predict(X_test)))

rf_party = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
rf_party.fit(X_train, y_party_train)
pred_party = rf_party.predict(X_test)
print("Party prediction accuracy:", accuracy_score(y_party_test, pred_party))
print(classification_report(y_party_test, pred_party))

# ---------------------------------------------------------
# STEP 8: Save trained models + encoders
# ---------------------------------------------------------
joblib.dump(rf_margin, MODEL_DIR / 'rf_margin_model.pkl')
joblib.dump(rf_retained, MODEL_DIR / 'rf_retained_model.pkl')
joblib.dump(rf_party, MODEL_DIR / 'rf_party_model.pkl')
joblib.dump(state_encoder, MODEL_DIR / 'state_encoder.pkl')
joblib.dump(party_encoder, MODEL_DIR / 'party_encoder.pkl')

print("\nAll models and encoders saved to", MODEL_DIR)