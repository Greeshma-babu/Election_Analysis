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
#
# IMPORTANT:
# These MUST match the features used by election_model.py
# during model training.
#
# Order is also important.
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
        # Normalize incoming data.
        #
        # The application may provide:
        #
        # 1. Long format:
        #    constituency_id
        #    state
        #    demographic
        #    result
        #    voter_turnout_pct
        #    margin_of_victory_pct
        #    swing_factor_pct
        #    election_year
        #
        # OR
        #
        # 2. Wide / merged format:
        #    result_2011
        #    result_2016
        #    result_2021
        #    turnout_2011
        #    turnout_2016
        #    turnout_2021
        #
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

        # -------------------------------------------------
        # Already wide / merged
        # -------------------------------------------------

        if not long_format:

            return self._clean_wide_data(data)

        # -------------------------------------------------
        # Clean election year
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
        # Create yearly frames
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

                "result":
                    f"result_{year}",

                "voter_turnout_pct":
                    f"turnout_{year}",

                "margin_of_victory_pct":
                    f"margin_{year}",

                "swing_factor_pct":
                    f"swing_{year}"
            }

            year_df = year_df.rename(
                columns=rename_map
            )

            # -------------------------------------------------
            # Keep useful columns
            # -------------------------------------------------

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

            # -------------------------------------------------
            # State/demographic already exist from first
            # dataset.
            # -------------------------------------------------

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
        # Recover state if missing
        # -------------------------------------------------

        if (
            "state" not in merged.columns
            and "state" in data.columns
        ):

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

        # -------------------------------------------------
        # Recover demographic if missing
        # -------------------------------------------------

        if (
            "demographic" not in merged.columns
            and "demographic" in data.columns
        ):

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
        # Clean final merged data
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
        # Alternate column names
        # -------------------------------------------------

        alternate_map = {

            "voter_turnout_2011_pct":
                "turnout_2011",

            "voter_turnout_2016_pct":
                "turnout_2016",

            "voter_turnout_2021_pct":
                "turnout_2021",

            "margin_of_victory_2011_pct":
                "margin_2011",

            "margin_of_victory_2016_pct":
                "margin_2016",

            "margin_of_victory_2021_pct":
                "margin_2021",

            "swing_factor_2011_pct":
                "swing_2011",

            "swing_factor_2016_pct":
                "swing_2016",

            "swing_factor_2021_pct":
                "swing_2021"
        }

        for old_column, new_column in alternate_map.items():

            if (
                old_column in data.columns
                and new_column not in data.columns
            ):

                data[new_column] = data[
                    old_column
                ]

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
        # Required model files
        # -------------------------------------------------

        required_models = [

            "rf_margin_model.pkl",
            "rf_retained_model.pkl",
            "rf_party_model.pkl"
        ]

        missing_models = [

            filename

            for filename in required_models

            if not (
                MODEL_DIR / filename
            ).exists()
        ]

        if missing_models:

            raise FileNotFoundError(

                "Missing model file(s): "
                + ", ".join(missing_models)
                + f"\nExpected location: {MODEL_DIR}"
            )

        # -------------------------------------------------
        # Load margin model
        # -------------------------------------------------

        self.rf_margin_model = joblib.load(
            MODEL_DIR / "rf_margin_model.pkl"
        )

        # -------------------------------------------------
        # Load retention model
        # -------------------------------------------------

        self.rf_retained_model = joblib.load(
            MODEL_DIR / "rf_retained_model.pkl"
        )

        # -------------------------------------------------
        # Load party model
        # -------------------------------------------------

        self.rf_party_model = joblib.load(
            MODEL_DIR / "rf_party_model.pkl"
        )

        # -------------------------------------------------
        # Load state encoder JSON
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
        # JSON values may be numbers represented as int.
        # Convert explicitly.
        # -------------------------------------------------

        self.state_map = {

            str(key): int(value)

            for key, value
            in self.state_map.items()
        }

        # -------------------------------------------------
        # Load party encoder JSON
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

        self.party_map = {

            str(key): int(value)

            for key, value
            in self.party_map.items()
        }

        self._models_loaded = True


    # =====================================================
    # BUILD MODEL FEATURES
    # =====================================================

    def _ensure_features(self):

        self._load_models()

        df = self.df.copy()

        # -------------------------------------------------
        # Required columns
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

        # =================================================
        # DEMOGRAPHIC ENCODING
        # =================================================

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
                + str(
                    list(
                        DEMOGRAPHIC_MAP.keys()
                    )
                )
            )

        # =================================================
        # STATE ENCODING
        # =================================================

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

        # =================================================
        # PARTY ENCODING
        # =================================================

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

        # =================================================
        # TURNOUT CHANGE
        # =================================================

        df["turnout_change_21"] = (

            df["turnout_2021"]
            - df["turnout_2016"]
        )

        # =================================================
        # NUMERIC CONVERSION
        # =================================================

        for column in FEATURE_COLS:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        # =================================================
        # CHECK MISSING FEATURES
        # =================================================

        missing_values = (

            df[
                FEATURE_COLS
            ]
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
    # GET 2011 TURNOUT COLUMN
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
    # GET LATITUDE COLUMN
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
    # GET LONGITUDE COLUMN
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
        # Historical chart does not require ML models.
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

            return

        # -------------------------------------------------
        # Election year
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
        # Count seats
        # -------------------------------------------------

        party_counts = (

            self.df[
                result_column
            ]

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
        # Bar chart
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

            textfont=dict(
                size=10
            )
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
    # TURNOUT SCENARIO SIMULATOR
    # =====================================================

    def plot_turnout_scenario(self):

        st.markdown(
            "### 🗳️ Turnout Scenario Simulator"
        )

        st.caption(
            "Change voter turnout and compare predicted election outcomes."
        )

        # -------------------------------------------------
        # Prepare ML features
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

                self.df[
                    "state"
                ]
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
        # BASELINE FEATURES
        # =================================================

        baseline = self.df[
            FEATURE_COLS
        ].copy()

        # =================================================
        # SCENARIO FEATURES
        # =================================================

        scenario = baseline.copy()

        scenario["turnout_2021"] = (

            scenario[
                "turnout_2021"
            ]
            + turnout_delta
        )

        scenario["turnout_2021"] = (

            scenario[
                "turnout_2021"
            ]
            .clip(0, 100)
        )

        scenario["turnout_change_21"] = (

            scenario[
                "turnout_2021"
            ]
            - scenario[
                "turnout_2016"
            ]
        )

        # =================================================
        # RESULT DATAFRAME
        # =================================================

        result = self.df.copy()

        # =================================================
        # MARGIN PREDICTION
        # =================================================

        result[
            "baseline_pred_margin"
        ] = (

            self.rf_margin_model.predict(
                baseline
            )
        )

        result[
            "scenario_pred_margin"
        ] = (

            self.rf_margin_model.predict(
                scenario
            )
        )

        # =================================================
        # RETENTION PREDICTION
        # =================================================

        result[
            "baseline_pred_retained"
        ] = (

            self.rf_retained_model.predict(
                baseline
            )
        )

        result[
            "scenario_pred_retained"
        ] = (

            self.rf_retained_model.predict(
                scenario
            )
        )

        # =================================================
        # PARTY PREDICTION
        # =================================================

        result[
            "baseline_pred_winner"
        ] = (

            self.rf_party_model.predict(
                baseline
            )
        )

        result[
            "scenario_pred_winner"
        ] = (

            self.rf_party_model.predict(
                scenario
            )
        )

        # =================================================
        # WINNER CHANGE
        # =================================================

        result[
            "winner_changed"
        ] = (

            result[
                "baseline_pred_winner"
            ]
            !=
            result[
                "scenario_pred_winner"
            ]
        )

        # =================================================
        # NEW TURNOUT
        # =================================================

        result[
            "new_turnout_2021"
        ] = (

            result[
                "turnout_2021"
            ]
            + turnout_delta
        ).clip(0, 100)

        # =================================================
        # RETENTION LABEL
        # =================================================

        result[
            "baseline_pred_retained_label"
        ] = (

            result[
                "baseline_pred_retained"
            ]
            .map({
                1: "Retained",
                0: "Lost"
            })
        )

        result[
            "scenario_pred_retained_label"
        ] = (

            result[
                "scenario_pred_retained"
            ]
            .map({
                1: "Retained",
                0: "Lost"
            })
        )

        # =================================================
        # FILTER STATES
        # =================================================

        result_filtered = result[

            result[
                "state"
            ].isin(
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
            ]
            .mean()
        )

        avg_after = (

            result_filtered[
                "scenario_pred_margin"
            ]
            .mean()
        )

        margin_change = (

            avg_after
            - avg_before
        )

        flips = int(

            result_filtered[
                "winner_changed"
            ]
            .sum()
        )

        avg_turnout = (

            result_filtered[
                "new_turnout_2021"
            ]
            .mean()
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

                color_discrete_map=
                    RETAINED_COLORS,

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

                    font=dict(
                        size=9
                    )
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

            changed_counts[
                "Changed"
            ] = (

                changed_counts[
                    "Changed"
                ]

                .map({

                    True:
                        "Party Flips",

                    False:
                        "Same Party"
                })
            )

            fig_pie2 = px.pie(

                changed_counts,

                names="Changed",

                values="Seats",

                color="Changed",

                color_discrete_map=
                    FLIP_COLORS,

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

                    font=dict(
                        size=9
                    )
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
                    result_filtered[
                        "state"
                    ] == state
                ]

                if turnout_2011_col:

                    trend_rows.append({

                        "State":
                            state,

                        "Election":
                            "2011",

                        "Turnout (%)":
                            state_df[
                                turnout_2011_col
                            ].mean()
                    })

                trend_rows.append({

                    "State":
                        state,

                    "Election":
                        "2016",

                    "Turnout (%)":
                        state_df[
                            "turnout_2016"
                        ].mean()
                })

                trend_rows.append({

                    "State":
                        state,

                    "Election":
                        "2021",

                    "Turnout (%)":
                        state_df[
                            "turnout_2021"
                        ].mean()
                })

                trend_rows.append({

                    "State":
                        state,

                    "Election":
                        "Scenario",

                    "Turnout (%)":
                        state_df[
                            "new_turnout_2021"
                        ].mean()
                })

            trend_df = pd.DataFrame(
                trend_rows
            )

            if trend_df.empty:

                st.info(
                    "No turnout trend data available."
                )

            else:

                fig_line = px.line(

                    trend_df,

                    x="Election",

                    y="Turnout (%)",

                    color="State",

                    markers=True
                )

                fig_line.update_traces(

                    line=dict(
                        width=2
                    ),

                    marker=dict(
                        size=6
                    )
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

                    font=dict(
                        size=9
                    ),

                    xaxis=dict(

                        tickangle=0,

                        tickfont=dict(
                            size=9
                        )
                    ),

                    legend=dict(

                        orientation="h",

                        y=-0.25,

                        x=0,

                        font=dict(
                            size=8
                        )
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
            .copy()
        )

        numeric_display_columns = [

            "baseline_pred_margin",

            "scenario_pred_margin"
        ]

        for column in numeric_display_columns:

            if column in display_df.columns:

                display_df[column] = display_df[
                    column
                ].round(2)

        rename_map = {

            "constituency_id":
                "ID",

            "state":
                "State",

            "baseline_pred_margin":
                "Margin Before %",

            "scenario_pred_margin":
                "Margin After %",

            "baseline_pred_winner":
                "Winner Before",

            "scenario_pred_winner":
                "Winner After",

            "scenario_pred_retained_label":
                "Status After"
        }

        display_df = display_df.rename(
            columns=rename_map
        )

        st.dataframe(

            display_df,

            width="stretch",

            height=260,

            hide_index=True
        )

        # -------------------------------------------------
        # Return the result so that app.py can optionally
        # use the prediction dataframe for another chart/map.
        # -------------------------------------------------

        return result_filtered


    # =====================================================
    # MERGED ELECTION MAP
    # =====================================================

    def plot_merged_election_map(
        self,
        prediction_df=None,
        turnout_delta=0
    ):

        st.markdown(
            "### 🗺️ Constituency Election Map"
        )

        st.caption(
            "Merged view of constituency results for "
            "2011, 2016, 2021 and the turnout-based prediction."
        )

        # =================================================
        # REQUIRED COLUMNS
        # =================================================

        required_columns = [

            "constituency_id",

            "state",

            "result_2011",
            "result_2016",
            "result_2021",

            "turnout_2011",
            "turnout_2016",
            "turnout_2021"
        ]

        missing = [

            column

            for column in required_columns

            if column not in self.df.columns
        ]

        if missing:

            st.error(

                "Required election columns are missing: "
                + ", ".join(missing)
            )

            return

        # =================================================
        # MAP VIEW SELECTOR
        # =================================================

        map_year = st.selectbox(

            "Map View",

            [
                "2011",
                "2016",
                "2021",
                "New Prediction"
            ],

            key="merged_map_year"
        )

        # =================================================
        # BASE DATA
        # =================================================

        map_df = self.df.copy()

        # =================================================
        # SELECT HISTORICAL YEAR
        # =================================================

        if map_year == "2011":

            map_df["map_winner"] = (
                map_df["result_2011"]
            )

            map_df["map_turnout"] = (
                map_df["turnout_2011"]
            )

            map_title = (
                "2011 Election Result"
            )

        elif map_year == "2016":

            map_df["map_winner"] = (
                map_df["result_2016"]
            )

            map_df["map_turnout"] = (
                map_df["turnout_2016"]
            )

            map_title = (
                "2016 Election Result"
            )

        elif map_year == "2021":

            map_df["map_winner"] = (
                map_df["result_2021"]
            )

            map_df["map_turnout"] = (
                map_df["turnout_2021"]
            )

            map_title = (
                "2021 Election Result"
            )

        # =================================================
        # NEW PREDICTION
        # =================================================

        else:

            if prediction_df is None:

                st.info(
                    "Run the Turnout Scenario Simulator "
                    "to generate the new prediction."
                )

                return

            map_df = prediction_df.copy()

            if (
                "scenario_pred_winner"
                not in map_df.columns
            ):

                st.warning(
                    "Prediction winner column is not available."
                )

                return

            map_df["map_winner"] = (
                map_df[
                    "scenario_pred_winner"
                ]
            )

            if (
                "new_turnout_2021"
                in map_df.columns
            ):

                map_df["map_turnout"] = (
                    map_df[
                        "new_turnout_2021"
                    ]
                )

            else:

                map_df["map_turnout"] = (
                    map_df[
                        "turnout_2021"
                    ]
                    + turnout_delta
                ).clip(0, 100)

            map_title = (

                "New Prediction "
                f"(Turnout {turnout_delta:+d}%)"
            )

        # =================================================
        # DISPLAY LABELS
        # =================================================

        map_df["Constituency"] = (

            map_df[
                "constituency_id"
            ]
            .astype(str)
        )

        map_df["State"] = (

            map_df[
                "state"
            ]
            .astype(str)
        )

        map_df["Winner"] = (

            map_df[
                "map_winner"
            ]
            .astype(str)
        )

        # =================================================
        # HISTORICAL INFORMATION
        # =================================================

        map_df["2011 Winner"] = (
            map_df["result_2011"]
        )

        map_df["2016 Winner"] = (
            map_df["result_2016"]
        )

        map_df["2021 Winner"] = (
            map_df["result_2021"]
        )

        map_df["2011 Turnout"] = (
            map_df["turnout_2011"]
        )

        map_df["2016 Turnout"] = (
            map_df["turnout_2016"]
        )

        map_df["2021 Turnout"] = (
            map_df["turnout_2021"]
        )

        # =================================================
        # PREDICTION INFORMATION
        # =================================================

        if prediction_df is not None:

            if (
                "scenario_pred_winner"
                in map_df.columns
            ):

                map_df[
                    "Predicted Winner"
                ] = map_df[
                    "scenario_pred_winner"
                ]

            if (
                "scenario_pred_margin"
                in map_df.columns
            ):

                map_df[
                    "Predicted Margin"
                ] = map_df[
                    "scenario_pred_margin"
                ]

            if (
                "scenario_pred_retained_label"
                in map_df.columns
            ):

                map_df[
                    "Prediction Status"
                ] = map_df[
                    "scenario_pred_retained_label"
                ]

            if (
                "new_turnout_2021"
                in map_df.columns
            ):

                map_df[
                    "Scenario Turnout"
                ] = map_df[
                    "new_turnout_2021"
                ]

        # =================================================
        # PARTY COLORS
        # =================================================

        parties = sorted(

            map_df[
                "Winner"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        palette = [

            "#E74C3C",
            "#3498DB",
            "#2ECC71",
            "#F39C12",
            "#9B59B6",
            "#1ABC9C",
            "#E67E22",
            "#34495E",
            "#D35400",
            "#8E44AD",
            "#C0392B",
            "#16A085"
        ]

        party_colors = {

            party:
                palette[
                    i % len(palette)
                ]

            for i, party
            in enumerate(parties)
        }

        # =================================================
        # COLOR COLUMN
        # =================================================

        map_df["Color"] = (

            map_df[
                "Winner"
            ]
            .map(party_colors)
            .fillna("#7F8C8D")
        )

        # =================================================
        # CREATE NON-GEOGRAPHIC CONSTITUENCY MAP
        # =================================================
        #
        # IMPORTANT:
        #
        # This is intentionally NOT using latitude /
        # longitude.
        #
        # Constituencies are displayed by State + ID.
        #
        # =================================================

        fig = px.scatter(

            map_df,

            x="State",

            y="Constituency",

            color="Winner",

            color_discrete_map=party_colors,

            hover_name="Constituency",

            hover_data={

                "State": True,

                "Winner": True,

                "2011 Winner": True,

                "2016 Winner": True,

                "2021 Winner": True,

                "2011 Turnout": ":.2f",

                "2016 Turnout": ":.2f",

                "2021 Turnout": ":.2f",

                "Constituency": False
            },

            title=map_title
        )

        # =================================================
        # PREDICTION HOVER
        # =================================================

        if (

            prediction_df is not None

            and "Predicted Winner"
            in map_df.columns
        ):

            hover_template = (

                "<b>%{customdata[0]}</b><br>"

                "State: %{customdata[1]}<br>"

                "Winner: %{customdata[2]}<br>"

                "<br>"

                "<b>Historical</b><br>"

                "2011: %{customdata[3]}<br>"

                "2016: %{customdata[4]}<br>"

                "2021: %{customdata[5]}<br>"

                "<br>"

                "<b>Turnout</b><br>"

                "2011: %{customdata[6]:.2f}%<br>"

                "2016: %{customdata[7]:.2f}%<br>"

                "2021: %{customdata[8]:.2f}%<br>"
            )

            if (
                "Predicted Margin"
                in map_df.columns
            ):

                hover_template += (

                    "<br>"

                    "<b>Prediction</b><br>"

                    "Winner: %{customdata[9]}<br>"

                    "Margin: %{customdata[10]:.2f}%<br>"
                )

            fig.update_traces(

                customdata=map_df[

                    [

                        "Constituency",
                        "State",
                        "Winner",

                        "2011 Winner",
                        "2016 Winner",
                        "2021 Winner",

                        "2011 Turnout",
                        "2016 Turnout",
                        "2021 Turnout",

                        "Predicted Winner",
                        "Predicted Margin"
                    ]

                ].values,

                hovertemplate=hover_template
            )

        else:

            fig.update_traces(

                marker=dict(
                    size=10
                )
            )

        # =================================================
        # LAYOUT
        # =================================================

        fig.update_layout(

            height=650,

            margin=dict(

                l=20,

                r=20,

                t=50,

                b=80
            ),

            xaxis_title="State",

            yaxis_title="Constituency",

            font=dict(
                size=10
            ),

            legend=dict(

                orientation="h",

                y=-0.15,

                x=0
            )
        )

        fig.update_xaxes(
            tickangle=-30
        )

        # =================================================
        # DISPLAY
        # =================================================

        st.plotly_chart(

            fig,

            width="stretch",

            config={
                "displayModeBar": False
            }
        )

        # =================================================
        # SUMMARY
        # =================================================

        st.markdown(

            f"**{map_title} — "
            f"{len(map_df)} constituencies**"
        )

        # =================================================
        # PARTY SUMMARY
        # =================================================

        summary = (

            map_df[
                "Winner"
            ]

            .value_counts()

            .reset_index()
        )

        summary.columns = [

            "Party",
            "Seats"
        ]

        col1, col2 = st.columns(2)

        with col1:

            st.dataframe(

                summary,

                width="stretch",

                hide_index=True
            )

        with col2:

            turnout_values = [

                map_df[
                    "turnout_2011"
                ].mean(),

                map_df[
                    "turnout_2016"
                ].mean(),

                map_df[
                    "turnout_2021"
                ].mean()
            ]

            turnout_summary = pd.DataFrame({

                "Election": [

                    "2011",
                    "2016",
                    "2021"
                ],

                "Average Turnout": (

                    turnout_values
                )
            })

            if (

                prediction_df is not None

                and "new_turnout_2021"
                in map_df.columns
            ):

                turnout_summary.loc[
                    len(turnout_summary)
                ] = [

                    "Scenario",

                    map_df[
                        "new_turnout_2021"
                    ].mean()
                ]

            turnout_summary[
                "Average Turnout"
            ] = turnout_summary[
                "Average Turnout"
            ].round(2)

            st.dataframe(

                turnout_summary,

                width="stretch",

                hide_index=True
            )


    # =====================================================
    # OPTIONAL FOLIUM MAP
    # =====================================================
    #
    # This method is kept available if you later provide
    # latitude/longitude data.
    #
    # It is NOT required by the current constituency map.
    # =====================================================

    def plot_geographic_map(self, prediction_df=None):

        st.markdown(
            "### 🌍 Geographic Election Map"
        )

        latitude_column = (
            self._get_latitude_column()
        )

        longitude_column = (
            self._get_longitude_column()
        )

        if (
            latitude_column is None
            or longitude_column is None
        ):

            st.info(

                "Latitude and longitude columns are not "
                "available. Geographic map is therefore "
                "not displayed."
            )

            return

        map_df = self.df.copy()

        map_df[latitude_column] = pd.to_numeric(
            map_df[latitude_column],
            errors="coerce"
        )

        map_df[longitude_column] = pd.to_numeric(
            map_df[longitude_column],
            errors="coerce"
        )

        map_df = map_df.dropna(
            subset=[
                latitude_column,
                longitude_column
            ]
        )

        if map_df.empty:

            st.warning(
                "No valid latitude/longitude records available."
            )

            return

        # -------------------------------------------------
        # Determine winner
        # -------------------------------------------------

        if prediction_df is not None:

            prediction_lookup = (

                prediction_df[
                    [
                        "constituency_id",
                        "scenario_pred_winner"
                    ]
                ]
                .drop_duplicates(
                    subset="constituency_id"
                )
            )

            map_df = map_df.merge(

                prediction_lookup,

                on="constituency_id",

                how="left"
            )

            map_df["Map Winner"] = (

                map_df[
                    "scenario_pred_winner"
                ]
                .fillna(
                    map_df["result_2021"]
                )
            )

        else:

            map_df["Map Winner"] = (
                map_df["result_2021"]
            )

        # -------------------------------------------------
        # Center
        # -------------------------------------------------

        center_lat = map_df[
            latitude_column
        ].mean()

        center_lon = map_df[
            longitude_column
        ].mean()

        folium_map = folium.Map(

            location=[
                center_lat,
                center_lon
            ],

            zoom_start=7
        )

        # -------------------------------------------------
        # Markers
        # -------------------------------------------------

        for _, row in map_df.iterrows():

            constituency = str(
                row["constituency_id"]
            )

            state = str(
                row.get(
                    "state",
                    ""
                )
            )

            winner = str(
                row["Map Winner"]
            )

            turnout = row.get(
                "turnout_2021",
                None
            )

            popup_text = (

                f"<b>Constituency:</b> "
                f"{constituency}<br>"

                f"<b>State:</b> "
                f"{state}<br>"

                f"<b>Winner:</b> "
                f"{winner}<br>"
            )

            if pd.notna(turnout):

                popup_text += (

                    f"<b>2021 Turnout:</b> "
                    f"{float(turnout):.2f}%"
                )

            folium.CircleMarker(

                location=[

                    row[
                        latitude_column
                    ],

                    row[
                        longitude_column
                    ]
                ],

                radius=6,

                popup=folium.Popup(
                    popup_text,
                    max_width=300
                ),

                fill=True
            ).add_to(
                folium_map
            )

        st_folium(

            folium_map,

            width=None,

            height=600
        )