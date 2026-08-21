import pathlib
import json
import joblib

import pandas as pd
import plotly.express as px
import streamlit as st

import folium
from streamlit_folium import st_folium


# =========================================================
# PATHS
# =========================================================

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# =========================================================
# MODEL FEATURES
# =========================================================

FEATURE_COLS = [
    "state_encoded",
    "demographic_encoded",
    "turnout_2016",
    "turnout_2021",
    "turnout_change_21",
    "margin_2016",
    "swing_2016",
    "result_2016_encoded"
]


# =========================================================
# DEMOGRAPHIC ENCODING
# =========================================================

DEMOGRAPHIC_MAP = {
    "Urban": 0,
    "Semi-Urban": 1,
    "Rural": 2
}


# =========================================================
# COLORS
# =========================================================

RETAINED_COLORS = {
    "Retained": "#2ECC71",
    "Lost": "#E74C3C"
}


FLIP_COLORS = {
    "Party Flips": "#F39C12",
    "Same Party": "#3498DB"
}


# =========================================================
# CSS
# =========================================================

def add_compact_css():

    st.markdown(
        """
        <style>

        .block-container {
            padding-top: 0.8rem;
            padding-bottom: 0.5rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        h1 {
            font-size: 1.6rem !important;
        }

        h2 {
            font-size: 1.25rem !important;
        }

        h3 {
            font-size: 1rem !important;
        }

        [data-testid="stMetric"] {
            padding: 0.15rem 0.3rem;
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.72rem !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.15rem !important;
        }

        [data-testid="stMetricDelta"] {
            font-size: 0.68rem !important;
        }

        [data-testid="stDataFrame"] {
            font-size: 0.72rem !important;
        }

        .stMultiSelect label,
        .stSelectbox label,
        .stSlider label {
            font-size: 0.75rem !important;
        }

        div[data-testid="stVerticalBlock"] {
            gap: 0.35rem;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# ELECTION CHARTS
# =========================================================

class ElectionCharts:

    def __init__(self, df):

        # -------------------------------------------------
        # IMPORTANT
        #
        # The application may pass either:
        #
        # 1. Long format:
        #
        # constituency_id
        # state
        # demographic
        # result
        # voter_turnout_pct
        # margin_of_victory_pct
        # swing_factor_pct
        # election_year
        #
        # OR
        #
        # 2. Already merged format:
        #
        # result_2011
        # result_2016
        # result_2021
        # turnout_2011
        # turnout_2016
        # turnout_2021
        #
        # Normalize everything here.
        # -------------------------------------------------

        self.df = self._prepare_election_data(df)

        self._models_loaded = False

        add_compact_css()


    # =====================================================
    # PREPARE ELECTION DATA
    # =====================================================

    def _prepare_election_data(self, df):

        if df is None:

            return pd.DataFrame()

        data = df.copy()

        # -------------------------------------------------
        # Clean column names
        # -------------------------------------------------

        data.columns = [
            str(column).strip()
            for column in data.columns
        ]

        # -------------------------------------------------
        # Detect long-format dataset
        # -------------------------------------------------

        long_format = (
            "election_year" in data.columns
            and "result" in data.columns
            and "constituency_id" in data.columns
        )

        if not long_format:

            # Already merged / wide format
            return self._clean_wide_data(data)

        # -------------------------------------------------
        # Clean year
        # -------------------------------------------------

        data["election_year"] = pd.to_numeric(
            data["election_year"],
            errors="coerce"
        )

        data = data[
            data["election_year"].isin(
                [2011, 2016, 2021]
            )
        ].copy()

        # -------------------------------------------------
        # Clean strings
        # -------------------------------------------------

        for column in [
            "constituency_id",
            "state",
            "demographic",
            "result"
        ]:

            if column in data.columns:

                data[column] = (
                    data[column]
                    .astype(str)
                    .str.strip()
                )

        # -------------------------------------------------
        # Numeric columns
        # -------------------------------------------------

        numeric_columns = [
            "voter_turnout_pct",
            "margin_of_victory_pct",
            "swing_factor_pct"
        ]

        for column in numeric_columns:

            if column in data.columns:

                data[column] = pd.to_numeric(
                    data[column],
                    errors="coerce"
                )

        # -------------------------------------------------
        # Keep useful base columns
        # -------------------------------------------------

        base_columns = [
            "constituency_id",
            "state",
            "demographic"
        ]

        base_columns = [
            column
            for column in base_columns
            if column in data.columns
        ]

        # -------------------------------------------------
        # Create yearly wide datasets
        # -------------------------------------------------

        yearly_frames = []

        for year in [2011, 2016, 2021]:

            year_df = data[
                data["election_year"] == year
            ].copy()

            if year_df.empty:
                continue

            # -------------------------------------------------
            # Remove duplicate constituency rows
            # -------------------------------------------------

            year_df = year_df.drop_duplicates(
                subset=["constituency_id"]
            )

            # -------------------------------------------------
            # Rename yearly columns
            # -------------------------------------------------

            rename_map = {
                "result": f"result_{year}",
                "voter_turnout_pct": f"turnout_{year}",
                "margin_of_victory_pct": f"margin_{year}",
                "swing_factor_pct": f"swing_{year}"
            }

            year_df = year_df.rename(
                columns=rename_map
            )

            keep_columns = [
                "constituency_id"
            ]

            for column in [
                "state",
                "demographic",
                f"result_{year}",
                f"turnout_{year}",
                f"margin_{year}",
                f"swing_{year}"
            ]:

                if column in year_df.columns:

                    keep_columns.append(column)

            year_df = year_df[
                keep_columns
            ]

            yearly_frames.append(
                year_df
            )

        # -------------------------------------------------
        # No yearly data
        # -------------------------------------------------

        if not yearly_frames:

            return pd.DataFrame()

        # -------------------------------------------------
        # Merge all years
        # -------------------------------------------------

        merged = yearly_frames[0].copy()

        for next_df in yearly_frames[1:]:

            # Do not duplicate state/demographic
            duplicate_base = [
                column
                for column in [
                    "state",
                    "demographic"
                ]
                if column in next_df.columns
            ]

            next_df = next_df.drop(
                columns=duplicate_base,
                errors="ignore"
            )

            merged = merged.merge(
                next_df,
                on="constituency_id",
                how="outer"
            )

        # -------------------------------------------------
        # If state/demographic missing from first year,
        # recover them from original data.
        # -------------------------------------------------

        if "state" not in merged.columns:

            state_lookup = (
                data[
                    [
                        "constituency_id",
                        "state"
                    ]
                ]
                .drop_duplicates(
                    subset="constituency_id"
                )
            )

            merged = merged.merge(
                state_lookup,
                on="constituency_id",
                how="left"
            )

        if "demographic" not in merged.columns:

            demographic_lookup = (
                data[
                    [
                        "constituency_id",
                        "demographic"
                    ]
                ]
                .drop_duplicates(
                    subset="constituency_id"
                )
            )

            merged = merged.merge(
                demographic_lookup,
                on="constituency_id",
                how="left"
            )

        # -------------------------------------------------
        # Clean final data
        # -------------------------------------------------

        merged = self._clean_wide_data(
            merged
        )

        return merged


    # =====================================================
    # CLEAN WIDE DATA
    # =====================================================

    def _clean_wide_data(self, data):

        data = data.copy()

        data.columns = [
            str(column).strip()
            for column in data.columns
        ]

        # -------------------------------------------------
        # Strip strings
        # -------------------------------------------------

        for column in data.select_dtypes(
            include="object"
        ).columns:

            data[column] = (
                data[column]
                .astype(str)
                .str.strip()
            )

        # -------------------------------------------------
        # Numeric election columns
        # -------------------------------------------------

        numeric_columns = [
            "turnout_2011",
            "turnout_2016",
            "turnout_2021",
            "margin_2011",
            "margin_2016",
            "margin_2021",
            "swing_2011",
            "swing_2016",
            "swing_2021"
        ]

        # Support alternate original column names too

        alternate_map = {
            "voter_turnout_2011_pct": "turnout_2011",
            "voter_turnout_2016_pct": "turnout_2016",
            "voter_turnout_2021_pct": "turnout_2021",

            "margin_of_victory_2011_pct": "margin_2011",
            "margin_of_victory_2016_pct": "margin_2016",
            "margin_of_victory_2021_pct": "margin_2021",

            "swing_factor_2011_pct": "swing_2011",
            "swing_factor_2016_pct": "swing_2016",
            "swing_factor_2021_pct": "swing_2021"
        }

        for old_column, new_column in alternate_map.items():

            if (
                old_column in data.columns
                and new_column not in data.columns
            ):

                data[new_column] = data[
                    old_column
                ]

        for column in numeric_columns:

            if column in data.columns:

                data[column] = pd.to_numeric(
                    data[column],
                    errors="coerce"
                )

        # -------------------------------------------------
        # Remove duplicate constituencies
        # -------------------------------------------------

        if "constituency_id" in data.columns:

            data = data.drop_duplicates(
                subset="constituency_id"
            )

        return data


    # =====================================================
    # LOAD MODELS
    # =====================================================

    def _load_models(self):

        if self._models_loaded:

            return

        # -------------------------------------------------
        # Check model files
        # -------------------------------------------------

        required_models = [
            "rf_margin_model.pkl",
            "rf_retained_model.pkl",
            "rf_party_model.pkl"
        ]

        missing_models = [
            filename
            for filename in required_models
            if not (MODEL_DIR / filename).exists()
        ]

        if missing_models:

            raise FileNotFoundError(
                "Missing model file(s): "
                + ", ".join(missing_models)
                + f"\nExpected location: {MODEL_DIR}"
            )

        # -------------------------------------------------
        # Load models
        # -------------------------------------------------

        self.rf_margin_model = joblib.load(
            MODEL_DIR / "rf_margin_model.pkl"
        )

        self.rf_retained_model = joblib.load(
            MODEL_DIR / "rf_retained_model.pkl"
        )

        self.rf_party_model = joblib.load(
            MODEL_DIR / "rf_party_model.pkl"
        )

        # -------------------------------------------------
        # State encoder
        # -------------------------------------------------

        state_encoder_file = (
            MODEL_DIR / "state_encoder.json"
        )

        if not state_encoder_file.exists():

            raise FileNotFoundError(
                f"Missing {state_encoder_file}"
            )

        with open(
            state_encoder_file,
            "r",
            encoding="utf-8"
        ) as f:

            self.state_map = json.load(f)

        # -------------------------------------------------
        # Party encoder
        # -------------------------------------------------

        party_encoder_file = (
            MODEL_DIR / "party_encoder.json"
        )

        if not party_encoder_file.exists():

            raise FileNotFoundError(
                f"Missing {party_encoder_file}"
            )

        with open(
            party_encoder_file,
            "r",
            encoding="utf-8"
        ) as f:

            self.party_map = json.load(f)

        self._models_loaded = True


    # =====================================================
    # BUILD MODEL FEATURES
    # =====================================================

    def _ensure_features(self):

        self._load_models()

        df = self.df.copy()

        # -------------------------------------------------
        # Required base columns
        # -------------------------------------------------

        required_base = [
            "state",
            "demographic",
            "result_2016",
            "turnout_2016",
            "turnout_2021",
            "margin_2016",
            "swing_2016"
        ]

        missing_base = [
            column
            for column in required_base
            if column not in df.columns
        ]

        if missing_base:

            raise ValueError(
                "Cannot build ML features. "
                "Missing columns: "
                + ", ".join(missing_base)
            )

        # -------------------------------------------------
        # DEMOGRAPHIC
        # -------------------------------------------------

        if "demographic_encoded" not in df.columns:

            df["demographic_encoded"] = (
                df["demographic"]
                .map(DEMOGRAPHIC_MAP)
            )

        unknown_demographic = (
            df.loc[
                df["demographic_encoded"].isna(),
                "demographic"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        if unknown_demographic:

            raise ValueError(
                "Unknown demographic value(s): "
                + str(unknown_demographic)
                + ". Expected: "
                + str(list(DEMOGRAPHIC_MAP.keys()))
            )

        # -------------------------------------------------
        # STATE
        # -------------------------------------------------

        if "state_encoded" not in df.columns:

            df["state_encoded"] = (
                df["state"]
                .map(self.state_map)
            )

        if df["state_encoded"].isnull().any():

            unknown = (
                df.loc[
                    df["state_encoded"].isnull(),
                    "state"
                ]
                .dropna()
                .unique()
                .tolist()
            )

            raise ValueError(
                "Unknown state(s) not seen during training: "
                + str(unknown)
            )

        # -------------------------------------------------
        # PARTY
        # -------------------------------------------------

        if "result_2016_encoded" not in df.columns:

            df["result_2016_encoded"] = (
                df["result_2016"]
                .map(self.party_map)
            )

        if df["result_2016_encoded"].isnull().any():

            unknown = (
                df.loc[
                    df["result_2016_encoded"].isnull(),
                    "result_2016"
                ]
                .dropna()
                .unique()
                .tolist()
            )

            raise ValueError(
                "Unknown party name(s) not seen during training: "
                + str(unknown)
            )

        # -------------------------------------------------
        # TURNOUT CHANGE
        # -------------------------------------------------

        df["turnout_change_21"] = (
            df["turnout_2021"]
            - df["turnout_2016"]
        )

        # -------------------------------------------------
        # Numeric conversion
        # -------------------------------------------------

        for column in FEATURE_COLS:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        # -------------------------------------------------
        # Missing feature values
        # -------------------------------------------------

        missing_values = (
            df[FEATURE_COLS]
            .isnull()
            .sum()
        )

        missing_features = (
            missing_values[
                missing_values > 0
            ]
        )

        if not missing_features.empty:

            raise ValueError(
                "Missing/invalid values in ML features:\n"
                + str(missing_features)
            )

        self.df = df


    # =====================================================
    # GET 2011 TURNOUT
    # =====================================================

    def _get_turnout_2011_column(self):

        possible_columns = [
            "turnout_2011",
            "voter_turnout_2011_pct"
        ]

        for column in possible_columns:

            if column in self.df.columns:

                return column

        return None


    # =====================================================
    # GET LATITUDE
    # =====================================================

    def _get_latitude_column(self):

        possible_columns = [
            "latitude",
            "lat",
            "Latitude",
            "LATITUDE"
        ]

        for column in possible_columns:

            if column in self.df.columns:

                return column

        return None


    # =====================================================
    # GET LONGITUDE
    # =====================================================

    def _get_longitude_column(self):

        possible_columns = [
            "longitude",
            "lon",
            "lng",
            "Longitude",
            "LONGITUDE"
        ]

        for column in possible_columns:

            if column in self.df.columns:

                return column

        return None


    # =====================================================
    # SEATS WON BY PARTY
    # =====================================================

    def plot_seats_won_by_party(self):

        st.markdown(
            "### 🏛️ Seats Won by Party"
        )

        # -------------------------------------------------
        # IMPORTANT FIX
        #
        # DO NOT call _ensure_features() here.
        #
        # This chart only needs historical result columns.
        # It does NOT need turnout_2011 or ML features.
        # -------------------------------------------------

        required_columns = [
            "result_2011",
            "result_2016",
            "result_2021"
        ]

        missing = [
            column
            for column in required_columns
            if column not in self.df.columns
        ]

        if missing:

            st.error(
                "Historical election result columns are missing: "
                + ", ".join(missing)
            )

            st.info(
                "The application expects 2011, 2016 and 2021 "
                "CSV files with the columns: result, election_year, "
                "constituency_id."
            )

            return

        # -------------------------------------------------
        # YEAR SELECTOR
        # -------------------------------------------------

        year = st.selectbox(
            "Election Year",
            [
                "2011",
                "2016",
                "2021"
            ],
            key="party_year"
        )

        result_column = (
            f"result_{year}"
        )

        # -------------------------------------------------
        # COUNT SEATS
        # -------------------------------------------------

        party_counts = (
            self.df[result_column]
            .dropna()
            .astype(str)
            .str.strip()
            .value_counts()
            .reset_index()
        )

        party_counts.columns = [
            "Party",
            "Seats"
        ]

        # -------------------------------------------------
        # BAR CHART
        # -------------------------------------------------

        fig = px.bar(
            party_counts,
            x="Party",
            y="Seats",
            color="Party",
            text="Seats"
        )

        fig.update_traces(
            textposition="outside",
            textfont=dict(size=10)
        )

        fig.update_layout(

            height=270,

            margin=dict(
                l=25,
                r=15,
                t=15,
                b=35
            ),

            xaxis_title=None,
            yaxis_title=None,

            legend=dict(
                orientation="h",
                y=-0.25,
                x=0
            ),

            font=dict(
                size=10
            )
        )

        st.plotly_chart(
            fig,
            width="stretch",
            config={
                "displayModeBar": False
            }
        )


    # =====================================================
    # TURNOUT SCENARIO
    # =====================================================

    def plot_turnout_scenario(self):

        st.markdown(
            "### 🗳️ Turnout Scenario Simulator"
        )

        st.caption(
            "Change voter turnout and compare predicted election outcomes."
        )

        # -------------------------------------------------
        # ONLY HERE DO WE NEED ML FEATURES
        # -------------------------------------------------

        try:

            self._ensure_features()

        except Exception as e:

            st.error(
                f"Unable to prepare ML features: {e}"
            )

            return

        # =================================================
        # CONTROLS
        # =================================================

        control_left, control_right = st.columns(
            [3, 1]
        )

        with control_left:

            all_states = sorted(
                self.df["state"]
                .dropna()
                .unique()
                .tolist()
            )

            selected_states = st.multiselect(

                "Select State(s)",

                options=all_states,

                default=all_states,

                key="state_filter"
            )

        with control_right:

            turnout_delta = st.slider(

                "Turnout Change (%)",

                min_value=-10,

                max_value=20,

                value=5,

                step=1,

                key="turnout_delta_slider"
            )

        if not selected_states:

            st.warning(
                "Please select at least one state."
            )

            return

        # =================================================
        # BASELINE
        # =================================================

        baseline = self.df[
            FEATURE_COLS
        ].copy()

        # =================================================
        # SCENARIO
        # =================================================

        scenario = baseline.copy()

        scenario["turnout_2021"] = (

            scenario["turnout_2021"]
            + turnout_delta
        )

        # Keep turnout realistic
        scenario["turnout_2021"] = (
            scenario["turnout_2021"]
            .clip(0, 100)
        )

        scenario["turnout_change_21"] = (

            scenario["turnout_2021"]

            - scenario["turnout_2016"]
        )

        # =================================================
        # PREDICTIONS
        # =================================================

        result = self.df.copy()

        # -------------------------------------------------
        # MARGIN
        # -------------------------------------------------

        result["baseline_pred_margin"] = (

            self.rf_margin_model.predict(
                baseline
            )
        )

        result["scenario_pred_margin"] = (

            self.rf_margin_model.predict(
                scenario
            )
        )

        # -------------------------------------------------
        # RETENTION
        # -------------------------------------------------

        result["baseline_pred_retained"] = (

            self.rf_retained_model.predict(
                baseline
            )
        )

        result["scenario_pred_retained"] = (

            self.rf_retained_model.predict(
                scenario
            )
        )

        # -------------------------------------------------
        # PARTY
        # -------------------------------------------------

        result["baseline_pred_winner"] = (

            self.rf_party_model.predict(
                baseline
            )
        )

        result["scenario_pred_winner"] = (

            self.rf_party_model.predict(
                scenario
            )
        )

        # -------------------------------------------------
        # WINNER CHANGED
        # -------------------------------------------------

        result["winner_changed"] = (

            result["baseline_pred_winner"]

            != result["scenario_pred_winner"]
        )

        # -------------------------------------------------
        # NEW TURNOUT
        # -------------------------------------------------

        result["new_turnout_2021"] = (

            result["turnout_2021"]

            + turnout_delta
        ).clip(0, 100)

        # =================================================
        # RETENTION LABEL
        # =================================================

        result[
            "baseline_pred_retained_label"
        ] = result[
            "baseline_pred_retained"
        ].map({

            1: "Retained",

            0: "Lost"
        })

        result[
            "scenario_pred_retained_label"
        ] = result[
            "scenario_pred_retained"
        ].map({

            1: "Retained",

            0: "Lost"
        })

        # =================================================
        # FILTER STATES
        # =================================================

        result_filtered = result[
            result["state"].isin(
                selected_states
            )
        ].copy()

        if result_filtered.empty:

            st.warning(
                "No constituency data found for the selected state(s)."
            )

            return

        # =================================================
        # METRICS
        # =================================================

        avg_before = (
            result_filtered[
                "baseline_pred_margin"
            ].mean()
        )

        avg_after = (
            result_filtered[
                "scenario_pred_margin"
            ].mean()
        )

        margin_change = (
            avg_after
            - avg_before
        )

        flips = int(
            result_filtered[
                "winner_changed"
            ].sum()
        )

        avg_turnout = (
            result_filtered[
                "new_turnout_2021"
            ].mean()
        )

        # =================================================
        # METRIC ROW
        # =================================================

        m1, m2, m3, m4 = st.columns(4)

        with m1:

            st.metric(
                "Margin Before",
                f"{avg_before:.2f}%"
            )

        with m2:

            st.metric(
                "Margin After",
                f"{avg_after:.2f}%",
                delta=f"{margin_change:+.2f}%"
            )

        with m3:

            st.metric(
                "Party Flips",
                flips
            )

        with m4:

            st.metric(
                "New Avg Turnout",
                f"{avg_turnout:.2f}%",
                delta=f"{turnout_delta:+d}%"
            )

        # =================================================
        # THREE CHARTS
        # =================================================

        chart1, chart2, chart3 = st.columns(
            [1, 1, 1]
        )

        # =================================================
        # CHART 1 - RETENTION
        # =================================================

        with chart1:

            st.markdown(
                "#### 🟢 Retention"
            )

            retained_counts = (

                result_filtered[
                    "scenario_pred_retained_label"
                ]

                .value_counts()

                .reset_index()
            )

            retained_counts.columns = [
                "Status",
                "Seats"
            ]

            fig_pie1 = px.pie(

                retained_counts,

                names="Status",

                values="Seats",

                color="Status",

                color_discrete_map=RETAINED_COLORS,

                hole=0.50
            )

            fig_pie1.update_traces(

                textinfo="percent",

                textfont_size=9
            )

            fig_pie1.update_layout(

                height=220,

                margin=dict(
                    l=5,
                    r=5,
                    t=5,
                    b=5
                ),

                legend=dict(
                    orientation="h",
                    y=-0.05,
                    x=0,
                    font=dict(size=9)
                )
            )

            st.plotly_chart(

                fig_pie1,

                width="stretch",

                config={
                    "displayModeBar": False
                }
            )

            st.caption(
                "Predicted seats retained after turnout change."
            )

        # =================================================
        # CHART 2 - PARTY FLIPS
        # =================================================

        with chart2:

            st.markdown(
                "#### 🟠 Party Flips"
            )

            changed_counts = (

                result_filtered[
                    "winner_changed"
                ]

                .value_counts()

                .reset_index()
            )

            changed_counts.columns = [
                "Changed",
                "Seats"
            ]

            changed_counts["Changed"] = (

                changed_counts["Changed"]

                .map({

                    True: "Party Flips",

                    False: "Same Party"
                })
            )

            fig_pie2 = px.pie(

                changed_counts,

                names="Changed",

                values="Seats",

                color="Changed",

                color_discrete_map=FLIP_COLORS,

                hole=0.50
            )

            fig_pie2.update_traces(

                textinfo="percent",

                textfont_size=9
            )

            fig_pie2.update_layout(

                height=220,

                margin=dict(
                    l=5,
                    r=5,
                    t=5,
                    b=5
                ),

                legend=dict(
                    orientation="h",
                    y=-0.05,
                    x=0,
                    font=dict(size=9)
                )
            )

            st.plotly_chart(

                fig_pie2,

                width="stretch",

                config={
                    "displayModeBar": False
                }
            )

            st.caption(
                "Constituencies where predicted winner changes."
            )

        # =================================================
        # CHART 3 - TURNOUT TREND
        # =================================================

        with chart3:

            st.markdown(
                "#### 📈 Turnout Trend"
            )

            turnout_2011_col = (
                self._get_turnout_2011_column()
            )

            trend_rows = []

            for state in selected_states:

                state_df = result_filtered[
                    result_filtered["state"] == state
                ]

                if turnout_2011_col:

                    trend_rows.append({

                        "State": state,

                        "Election": "2011",

                        "Turnout (%)":
                            state_df[
                                turnout_2011_col
                            ].mean()
                    })

                trend_rows.append({

                    "State": state,

                    "Election": "2016",

                    "Turnout (%)":
                        state_df[
                            "turnout_2016"
                        ].mean()
                })

                trend_rows.append({

                    "State": state,

                    "Election": "2021",

                    "Turnout (%)":
                        state_df[
                            "turnout_2021"
                        ].mean()
                })

                trend_rows.append({

                    "State": state,

                    "Election": "Scenario",

                    "Turnout (%)":
                        state_df[
                            "new_turnout_2021"
                        ].mean()
                })

            trend_df = pd.DataFrame(
                trend_rows
            )

            fig_line = px.line(

                trend_df,

                x="Election",

                y="Turnout (%)",

                color="State",

                markers=True
            )

            fig_line.update_traces(

                line=dict(width=2),

                marker=dict(size=6)
            )

            fig_line.update_layout(

                height=220,

                margin=dict(
                    l=5,
                    r=5,
                    t=5,
                    b=35
                ),

                xaxis_title=None,

                yaxis_title="Turnout %",

                font=dict(size=9),

                xaxis=dict(
                    tickangle=0,
                    tickfont=dict(size=9)
                ),

                legend=dict(
                    orientation="h",
                    y=-0.25,
                    x=0,
                    font=dict(size=8)
                )
            )

            st.plotly_chart(

                fig_line,

                width="stretch",

                config={
                    "displayModeBar": False
                }
            )

            st.caption(
                "Average voter turnout across selected states."
            )

        # =================================================
        # CONSTITUENCY DETAILS
        # =================================================

        st.markdown(
            "### 📋 Constituency Details"
        )

        st.caption(
            "Predicted margin, winner and retention status "
            "for each selected constituency."
        )

        display_cols = [

            "constituency_id",

            "state",

            "baseline_pred_margin",

            "scenario_pred_margin",

            "baseline_pred_winner",

            "scenario_pred_winner",

            "scenario_pred_retained_label"
        ]

        display_cols = [
            column
            for column in display_cols
            if column in result_filtered.columns
        ]

        display_df = (
            result_filtered[
                display_cols
            ]
            .round(2)
            .copy()
        )

        display_df.columns = [

            "ID",

            "State",

            "Margin Before %",

            "Margin After %",

            "Winner Before",

            "Winner After",

            "Status After"
        ][:len(display_df.columns)]

        st.dataframe(

            display_df,

            width="stretch",

            height=260,

            hide_index=True
        )
