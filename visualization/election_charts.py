import pathlib
import json
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


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
# ENCODING MAPS
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
    "Same Party Wins": "#3498DB"
}


# =========================================================
# SMALL CSS
# =========================================================

def add_compact_css():

    st.markdown(
        """
        <style>

        /* Reduce main page top/bottom spacing */
        .block-container {
            padding-top: 0.8rem;
            padding-bottom: 0.5rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        /* Smaller headings */
        h1 {
            font-size: 1.6rem !important;
        }

        h2 {
            font-size: 1.25rem !important;
        }

        h3 {
            font-size: 1rem !important;
        }

        /* Smaller metric cards */
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

        /* Smaller dataframe text */
        [data-testid="stDataFrame"] {
            font-size: 0.72rem !important;
        }

        /* Smaller selectbox / multiselect */
        .stMultiSelect label,
        .stSelectbox label,
        .stSlider label {
            font-size: 0.75rem !important;
        }

        /* Reduce vertical gap between Streamlit elements */
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

        self.df = df

        self._models_loaded = False

        add_compact_css()


    # =====================================================
    # LOAD MODELS
    # =====================================================

    def _load_models(self):

        if not self._models_loaded:

            self.rf_margin_model = joblib.load(
                MODEL_DIR / "rf_margin_model.pkl"
            )

            self.rf_retained_model = joblib.load(
                MODEL_DIR / "rf_retained_model.pkl"
            )

            self.rf_party_model = joblib.load(
                MODEL_DIR / "rf_party_model.pkl"
            )

            # JSON encoders
            with open(
                MODEL_DIR / "state_encoder.json",
                "r"
            ) as f:

                self.state_map = json.load(f)

            with open(
                MODEL_DIR / "party_encoder.json",
                "r"
            ) as f:

                self.party_map = json.load(f)

            self._models_loaded = True


    # =====================================================
    # BUILD MODEL FEATURES
    # =====================================================

    def _ensure_features(self):

        self._load_models()

        df = self.df.copy()

        # -----------------------------------------------
        # Demographic
        # -----------------------------------------------

        if "demographic_encoded" not in df.columns:

            df["demographic_encoded"] = (
                df["demographic"]
                .map(DEMOGRAPHIC_MAP)
            )

        # -----------------------------------------------
        # State
        # -----------------------------------------------

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
                    .unique()
                    .tolist()
                )

                raise ValueError(
                    f"Unknown state(s) not seen during training: {unknown}"
                )

        # -----------------------------------------------
        # 2016 Result
        # -----------------------------------------------

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
                    .unique()
                    .tolist()
                )

                raise ValueError(
                    f"Unknown party name(s) not seen during training: {unknown}"
                )

        # -----------------------------------------------
        # Turnout change
        # -----------------------------------------------

        if "turnout_change_21" not in df.columns:

            df["turnout_change_21"] = (
                df["turnout_2021"]
                - df["turnout_2016"]
            )

        # -----------------------------------------------
        # Check required features
        # -----------------------------------------------

        missing = [
            column
            for column in FEATURE_COLS
            if column not in df.columns
        ]

        if missing:

            raise ValueError(
                "Cannot build required features. "
                f"Missing columns: {missing}"
            )

        self.df = df


    # =====================================================
    # GET 2011 TURNOUT COLUMN
    # =====================================================

    def _get_turnout_2011_column(self):

        if "turnout_2011" in self.df.columns:

            return "turnout_2011"

        if "voter_turnout_2011_pct" in self.df.columns:

            return "voter_turnout_2011_pct"

        return None


    # =====================================================
    # SEATS WON BY PARTY
    # =====================================================

    def plot_seats_won_by_party(self):

        st.markdown(
            "### 🏛️ Seats Won by Party"
        )

        year = st.selectbox(
            "Election Year",
            ["2011", "2016", "2021"],
            key="party_year"
        )

        result_column = f"result_{year}"

        party_counts = (
            self.df[result_column]
            .value_counts()
            .reset_index()
        )

        party_counts.columns = [
            "Party",
            "Seats"
        ]

        fig = px.bar(
            party_counts,
            x="Party",
            y="Seats",
            color="Party",
            text="Seats"
        )

        # -----------------------------------------------
        # Smaller bar chart
        # -----------------------------------------------

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

        self._ensure_features()

        # =================================================
        # STATE + TURNOUT CONTROLS
        # =================================================

        control_left, control_right = st.columns(
            [3, 1]
        )

        with control_left:

            all_states = sorted(
                self.df["state"]
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
            scenario["turnout_2021"]
            + turnout_delta
        )

        scenario["turnout_change_21"] = (
            scenario["turnout_2021"]
            - scenario["turnout_2016"]
        )

        # =================================================
        # PREDICTIONS
        # =================================================

        result = self.df.copy()

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

        result["winner_changed"] = (
            result["baseline_pred_winner"]
            != result["scenario_pred_winner"]
        )

        result["new_turnout_2021"] = (
            result["turnout_2021"]
            + turnout_delta
        )

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
            avg_after - avg_before
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
        # COMPACT METRIC ROW
        # =================================================

        m1, m2, m3, m4 = st.columns(4)

        with m1:

            st.metric(
                "Margin Before",
                f"{avg_before:.2f}%"
            )

        with m2:

            # Streamlit automatically shows:
            # green = increase
            # red   = decrease
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
        # THREE CHARTS IN ONE ROW
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
            "Predicted margin, winner and retention status for each selected constituency."
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
        ]

        # =================================================
        # COMPACT TABLE
        # =================================================

        st.dataframe(
            display_df,
            width="stretch",
            height=260,

            hide_index=True,

            column_config={

                "ID":
                    st.column_config.TextColumn(
                        "ID",
                        width="small"
                    ),

                "State":
                    st.column_config.TextColumn(
                        "State",
                        width="small"
                    ),

                "Margin Before %":
                    st.column_config.NumberColumn(
                        "Before %",
                        format="%.2f",
                        width="small"
                    ),

                "Margin After %":
                    st.column_config.NumberColumn(
                        "After %",
                        format="%.2f",
                        width="small"
                    ),

                "Winner Before":
                    st.column_config.TextColumn(
                        "Winner Before",
                        width="small"
                    ),

                "Winner After":
                    st.column_config.TextColumn(
                        "Winner After",
                        width="small"
                    ),

                "Status After":
                    st.column_config.TextColumn(
                        "Status",
                        width="small"
                    )
            }
        )