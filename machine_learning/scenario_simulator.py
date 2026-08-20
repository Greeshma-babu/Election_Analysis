import pathlib
import pandas as pd
import joblib

# ---------------------------------------------------------
# Resolve all paths relative to THIS script's own location,
# not the terminal's current directory - works no matter
# where you run it from.
# ---------------------------------------------------------
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent   # up from machine_learning/ to Election_Analysis/
MODEL_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'dataset' / 'training'

# ---------------------------------------------------------
# Load all three trained models + the encoders used during training
# ---------------------------------------------------------
rf_margin_model = joblib.load(MODEL_DIR / 'rf_margin_model.pkl')
rf_retained_model = joblib.load(MODEL_DIR / 'rf_retained_model.pkl')
rf_party_model = joblib.load(MODEL_DIR / 'rf_party_model.pkl')
state_encoder = joblib.load(MODEL_DIR / 'state_encoder.pkl')
party_encoder = joblib.load(MODEL_DIR / 'party_encoder.pkl')

# Must match election_model.py's feature_cols EXACTLY - same columns, same order
FEATURE_COLS = ['state_encoded', 'demographic_encoded', 'turnout_2016', 'turnout_2021',
                 'turnout_change_21', 'margin_2016', 'swing_2016', 'result_2016_encoded']


def simulate_turnout_change(df, turnout_delta_pct, margin_model, retained_model, party_model):
    baseline = df[FEATURE_COLS].copy()
    scenario = baseline.copy()
    scenario['turnout_2021'] = scenario['turnout_2021'] + turnout_delta_pct
    scenario['turnout_change_21'] = scenario['turnout_2021'] - scenario['turnout_2016']

    out = df.copy()
    out['baseline_pred_margin'] = margin_model.predict(baseline)
    out['scenario_pred_margin'] = margin_model.predict(scenario)

    out['baseline_pred_retained'] = retained_model.predict(baseline)
    out['scenario_pred_retained'] = retained_model.predict(scenario)

    out['baseline_pred_winner'] = party_model.predict(baseline)
    out['scenario_pred_winner'] = party_model.predict(scenario)
    out['winner_changed'] = out['baseline_pred_winner'] != out['scenario_pred_winner']

    return out


if __name__ == "__main__":
    X_test = pd.read_csv(DATA_DIR / 'X_test.csv')
    id_test = pd.read_csv(DATA_DIR / 'id_test.csv')

    result = simulate_turnout_change(X_test, 5, rf_margin_model, rf_retained_model, rf_party_model)
    result.insert(0, 'constituency_id', id_test['constituency_id'].values)  # attach IDs back for display

    result['baseline_pred_retained_label'] = result['baseline_pred_retained'].map({1: 'Retained', 0: 'Lost'})
    result['scenario_pred_retained_label'] = result['scenario_pred_retained'].map({1: 'Retained', 0: 'Lost'})

   
    
    print("=== REGRESSION: Margin of Victory (%) ===")
    print(result[['constituency_id', 'baseline_pred_margin', 'scenario_pred_margin']].to_string(index=False))

    print("\n=== CLASSIFICATION: Predicted Winner & Retention ===")
    print(result[[
        'constituency_id',
        'baseline_pred_winner', 'scenario_pred_winner',
        'baseline_pred_retained_label', 'scenario_pred_retained_label',
        'winner_changed'
    ]].to_string(index=False))

    result.to_csv(DATA_DIR / 'scenario_output.csv', index=False)
    print("\nFull results saved to", DATA_DIR / 'scenario_output.csv')