import altair as alt

"""
    _summary_
    This python code defines a function called get_chart that takes two arguments, data and chart_type, 
    and returns an Altair chart. The function uses the chart_type argument to determine which type of chart to 
    generate and returns the chart using the data argument as the data source. The chart includes hover-based interactivity, 
    where hovering over a data point will highlight the data point and display a tooltip with additional information. 
    The function supports two chart types, which are determined by the chart_type argument: a line chart showing average votes 
    per episode, and a line chart showing predicted votes for future episodes based on previous votes.
"""

def get_chart(data, chart_type):
    
    hover = alt.selection_single(
        fields=["episode"],
        nearest=True,
        on="mouseover",
        empty="none",
    )
    
    if chart_type == 0:
        
        lines = (
            alt.Chart(data, title="Avarage Vote Per Episode")
            .mark_line()
            .encode(
                x="episode",
                y="avg",
                color="season",
                strokeDash="season",
            )
        )

        # Draw points on the line, and highlight based on selection
        points = lines.transform_filter(hover).mark_circle(size=65)

        # Draw a rule at the location of the selection
        tooltips = (
            alt.Chart(data)
            .mark_rule()
            .encode(
                x="episode",
                y="avg",
                opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
                tooltip=[
                    alt.Tooltip("episode", title="Episode Number"),
                    alt.Tooltip("avg", title="AVG. Vote"),
                ],
            )
            .add_selection(hover)
        )
    elif chart_type == 1:
        
        lines = (
            alt.Chart(data, title="Predict Next Episode Vote Based On previous Votes")
            .mark_line()
            .encode(
                x="episode",
                y="Vote",
                color="Type",
                strokeDash="Type",
            )
        )

        points = lines.transform_filter(hover).mark_circle(size=60)

        tooltips = (
            alt.Chart(data)
            .mark_rule()
            .encode(
                x="episode",
                y="Vote",
                opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
                tooltip=[
                    alt.Tooltip("episode", title="Episode Number"),
                    alt.Tooltip("Vote", title="Vote"),
                ],
            )
            .add_selection(hover)
        )
        
    return (lines + points + tooltips).interactive()