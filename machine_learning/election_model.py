# =========================================================
# CONSTITUENCY ELECTION MAP
# DOES NOT REQUIRE LATITUDE / LONGITUDE
# =========================================================

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

    # =====================================================
    # REQUIRED COLUMNS
    # =====================================================

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
        col
        for col in required_columns
        if col not in self.df.columns
    ]

    if missing:

        st.error(
            "Required election columns are missing: "
            + ", ".join(missing)
        )

        return

    # =====================================================
    # MAP VIEW SELECTOR
    # =====================================================

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

    # =====================================================
    # BASE DATA
    # =====================================================

    if map_year == "2011":

        map_df = self.df.copy()

        map_df["map_winner"] = (
            map_df["result_2011"]
        )

        map_df["map_turnout"] = (
            map_df["turnout_2011"]
        )

        map_title = "2011 Election Result"

    elif map_year == "2016":

        map_df = self.df.copy()

        map_df["map_winner"] = (
            map_df["result_2016"]
        )

        map_df["map_turnout"] = (
            map_df["turnout_2016"]
        )

        map_title = "2016 Election Result"

    elif map_year == "2021":

        map_df = self.df.copy()

        map_df["map_winner"] = (
            map_df["result_2021"]
        )

        map_df["map_turnout"] = (
            map_df["turnout_2021"]
        )

        map_title = "2021 Election Result"

    else:

        # =================================================
        # NEW PREDICTION
        # =================================================

        if prediction_df is None:

            st.info(
                "Run the Turnout Scenario Simulator "
                "to generate the new prediction."
            )

            return

        map_df = prediction_df.copy()

        if "scenario_pred_winner" not in map_df.columns:

            st.warning(
                "Prediction winner column is not available."
            )

            return

        map_df["map_winner"] = (
            map_df["scenario_pred_winner"]
        )

        map_df["map_turnout"] = (
            map_df["new_turnout_2021"]
        )

        map_title = (
            f"New Prediction "
            f"(Turnout {turnout_delta:+d}%)"
        )

    # =====================================================
    # CREATE DISPLAY LABEL
    # =====================================================

    map_df["Constituency"] = (
        map_df["constituency_id"].astype(str)
    )

    map_df["State"] = (
        map_df["state"].astype(str)
    )

    map_df["Winner"] = (
        map_df["map_winner"].astype(str)
    )

    # =====================================================
    # HISTORICAL INFORMATION
    # =====================================================

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

    # =====================================================
    # PREDICTION INFORMATION
    # =====================================================

    if prediction_df is not None:

        if "scenario_pred_winner" in map_df.columns:

            map_df["Predicted Winner"] = (
                map_df["scenario_pred_winner"]
            )

        if "scenario_pred_margin" in map_df.columns:

            map_df["Predicted Margin"] = (
                map_df["scenario_pred_margin"]
            )

        if "scenario_pred_retained_label" in map_df.columns:

            map_df["Prediction Status"] = (
                map_df[
                    "scenario_pred_retained_label"
                ]
            )

        if "new_turnout_2021" in map_df.columns:

            map_df["Scenario Turnout"] = (
                map_df["new_turnout_2021"]
            )

    # =====================================================
    # PARTY COLORS
    # =====================================================

    parties = sorted(
        map_df["Winner"]
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
        party: palette[
            i % len(palette)
        ]
        for i, party in enumerate(parties)
    }

    # =====================================================
    # COLOR COLUMN
    # =====================================================

    map_df["Color"] = (
        map_df["Winner"]
        .map(party_colors)
        .fillna("#7F8C8D")
    )

    # =====================================================
    # STATE / CONSTITUENCY VISUAL MAP
    # =====================================================

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

    # =====================================================
    # PREDICTION HOVER DATA
    # =====================================================

    if (
        prediction_df is not None
        and "Predicted Winner" in map_df.columns
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

        if "Predicted Margin" in map_df.columns:

            hover_template += (
                "<br>"
                "<b>Prediction</b><br>"
                "Winner: %{customdata[9]}<br>"
                "Margin: %{customdata[10]:.2f}%<br>"
            )

        fig.update_traces(
            marker=dict(
                size=10
            )
        )

    else:

        fig.update_traces(
            marker=dict(
                size=10
            )
        )

    # =====================================================
    # LAYOUT
    # =====================================================

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

    # =====================================================
    # DISPLAY
    # =====================================================

    st.plotly_chart(

        fig,

        width="stretch",

        config={
            "displayModeBar": False
        }
    )

    # =====================================================
    # SUMMARY
    # =====================================================

    st.markdown(
        f"**{map_title} — {len(map_df)} constituencies**"
    )

    # =====================================================
    # PARTY SUMMARY
    # =====================================================

    summary = (

        map_df["Winner"]

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

        turnout_summary = pd.DataFrame({

            "Election": [
                "2011",
                "2016",
                "2021"
            ],

            "Average Turnout": [

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
        })

        if prediction_df is not None:

            if "new_turnout_2021" in map_df.columns:

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