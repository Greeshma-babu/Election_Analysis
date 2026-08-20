import plotly.express as px
import streamlit as st


class ElectionCharts:

    def __init__(self, df):
        self.df = df

    def plot_seats_won_by_party(self):

        st.subheader("Seats Won by Party")

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

        party_counts.columns = ["Party", "Seats"]

        fig = px.bar(
            party_counts,
            x="Party",
            y="Seats",
            color="Party",
            text="Seats",
            title=f"Seats by Party - {year}"
        )

        fig.update_layout(
            xaxis_title="Party",
            yaxis_title="Number of Constituencies",
            height=500
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )