import pathlib
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------
# Paths - this file lives in visualization/, so go up ONE
# level to reach the project root (Election_Analysis/)
# ---------------------------------------------------------
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / 'models'

# Must match election_model.py's feature_cols EXACTLY - same columns, same order
FEATURE_COLS = ['state_encoded', 'demographic_encoded', 'turnout_2016', 'turnout_2021',
                 'turnout_change_21', 'margin_2016', 'swing_2016', 'result_2016_encoded']

DEMOGRAPHIC_MAP = {'Urban': 0, 'Semi-Urban': 1, 'Rural': 2}

# Public India state-boundary geojson, used for the choropleth map
INDIA_STATES_GEOJSON_URL = "https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson"


class ElectionCharts:

    def __init__(self, df):
        self.df = df                # can be raw DB data OR the pre-engineered CSV - _ensure_features() handles both
        self._models_loaded = False

    def _load_models(self):
        """Load pretrained models + encoders only once, not on every chart redraw."""
        if not self._models_loaded:
            self.rf_margin_model = joblib.load(MODEL_DIR / 'rf_margin_model.pkl')
            self.rf_retained_model = joblib.load(MODEL_DIR / 'rf_retained_model.pkl')
            self.rf_party_model = joblib.load(MODEL_DIR / 'rf_party_model.pkl')
            self.state_encoder = joblib.load(MODEL_DIR / 'state_encoder.pkl')
            self.party_encoder = joblib.load(MODEL_DIR / 'party_encoder.pkl')
            self._models_loaded = True

    def _ensure_features(self):
        """
        Builds any FEATURE_COLS missing from self.df, using the SAME
        encoders fit during training - so encoded numbers match exactly
        what the models learned. Safe to call repeatedly; skips columns
        that already exist. Call this before any prediction-based chart.
        """
        self._load_models()
        df = self.df

        if 'demographic_encoded' not in df.columns:
            df['demographic_encoded'] = df['demographic'].map(DEMOGRAPHIC_MAP)

        if 'state_encoded' not in df.columns:
            df['state_encoded'] = self.state_encoder.transform(df['state'])

        if 'result_2016_encoded' not in df.columns:
            df['result_2016_encoded'] = self.party_encoder.transform(df['result_2016'])

        if 'turnout_change_21' not in df.columns:
            df['turnout_change_21'] = df['turnout_2021'] - df['turnout_2016']

        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Cannot build required features - these raw columns are missing "
                f"from the data entirely: {missing}. Check your DB schema / insert.sql."
            )

        self.df = df

    def plot_seats_won_by_party(self):
        st.subheader("Seats Won by Party")

        year = st.selectbox("Election Year", ["2011", "2016", "2021"], key="party_year")
        result_column = f"result_{year}"

        party_counts = self.df[result_column].value_counts().reset_index()
        party_counts.columns = ["Party", "Seats"]

        fig = px.bar(
            party_counts, x="Party", y="Seats", color="Party", text="Seats",
            title=f"Seats by Party - {year}"
        )
        fig.update_layout(xaxis_title="Party", yaxis_title="Number of Constituencies", height=500)
        st.plotly_chart(fig, width='stretch')

    def plot_turnout_scenario(self):
        """
        Lets the user drag a slider to change turnout by X%,
        runs the pretrained models on that scenario, and shows the result
        as a state-level map (and a bar chart fallback).
        """
        st.subheader("Turnout Change Scenario Simulator")

        self._ensure_features()   # builds state_encoded / demographic_encoded / etc. if missing

        turnout_delta = st.slider(
            "Change in Voter Turnout (%)",
            min_value=-10, max_value=20, value=5, step=1,
            key="turnout_delta_slider"
        )

        # Build baseline (current data) and scenario (turnout shifted) feature sets
        baseline = self.df[FEATURE_COLS].copy()
        scenario = baseline.copy()
        scenario['turnout_2021'] = scenario['turnout_2021'] + turnout_delta
        scenario['turnout_change_21'] = scenario['turnout_2021'] - scenario['turnout_2016']

        result = self.df.copy()
        result['baseline_pred_margin'] = self.rf_margin_model.predict(baseline)
        result['scenario_pred_margin'] = self.rf_margin_model.predict(scenario)
        result['baseline_pred_retained'] = self.rf_retained_model.predict(baseline)
        result['scenario_pred_retained'] = self.rf_retained_model.predict(scenario)
        result['baseline_pred_winner'] = self.rf_party_model.predict(baseline)
        result['scenario_pred_winner'] = self.rf_party_model.predict(scenario)
        result['winner_changed'] = result['baseline_pred_winner'] != result['scenario_pred_winner']

        # Top-line summary numbers
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Margin (baseline)", f"{result['baseline_pred_margin'].mean():.2f}%")
        col2.metric(
            "Avg Margin (scenario)",
            f"{result['scenario_pred_margin'].mean():.2f}%",
            delta=f"{(result['scenario_pred_margin'].mean() - result['baseline_pred_margin'].mean()):.2f}%"
        )
        col3.metric("Seats Flipping Winner", f"{result['winner_changed'].sum()} / {len(result)}")

        # Aggregate constituency-level predictions up to state level, for the map
        state_summary = (
            result.groupby('state')
            .agg(
                avg_baseline_margin=('baseline_pred_margin', 'mean'),
                avg_scenario_margin=('scenario_pred_margin', 'mean'),
                seats_flipped=('winner_changed', 'sum'),
                total_seats=('winner_changed', 'count')
            )
            .reset_index()
        )
        state_summary['flip_pct'] = (state_summary['seats_flipped'] / state_summary['total_seats'] * 100).round(1)

        tab1, tab2 = st.tabs(["Map View", "Bar Chart View"])

        with tab1:
            try:
                fig_map = px.choropleth(
                    state_summary,
                    geojson=INDIA_STATES_GEOJSON_URL,
                    featureidkey="properties.NAME_1",
                    locations="state",
                    color="avg_scenario_margin",
                    color_continuous_scale="RdYlGn",
                    hover_data=["avg_baseline_margin", "avg_scenario_margin", "seats_flipped", "flip_pct"],
                    title=f"Predicted Avg Margin by State (Turnout {'+' if turnout_delta >= 0 else ''}{turnout_delta}%)"
                )
                fig_map.update_geos(fitbounds="locations", visible=False)
                fig_map.update_layout(height=550)
                st.plotly_chart(fig_map, width='stretch')
            except Exception as e:
                st.warning(f"Map could not load (needs internet access to fetch India state boundaries): {e}")
                st.info("Use the 'Bar Chart View' tab instead.")

        with tab2:
            fig_bar = px.bar(
                state_summary,
                x="state",
                y=["avg_baseline_margin", "avg_scenario_margin"],
                barmode="group",
                title="Predicted Margin by State: Baseline vs Scenario",
                labels={"value": "Predicted Margin (%)", "state": "State", "variable": "Scenario"}
            )
            fig_bar.update_layout(height=500)
            st.plotly_chart(fig_bar, width='stretch')

        st.subheader("Seats Where Predicted Winner Changes")
        changed = result[result['winner_changed']][
            ['constituency_id', 'state', 'baseline_pred_winner', 'scenario_pred_winner',
             'baseline_pred_margin', 'scenario_pred_margin']
        ]
        st.dataframe(changed, width='stretch')