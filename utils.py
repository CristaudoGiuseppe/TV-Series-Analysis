import altair as alt, pandas as pd

def get_chart(data):
    hover = alt.selection_single(
        fields=["episode"],
        nearest=True,
        on="mouseover",
        empty="none",
    )

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

    return (lines + points + tooltips).interactive()

def window_input_output(input_length: int, output_length: int, data: pd.DataFrame) -> pd.DataFrame:
    
    df = data.copy()
    
    i = 1
    while i < input_length:
        df[f'x_{i}'] = df['co2'].shift(-i)
        i = i + 1
        
    j = 0
    while j < output_length:
        df[f'y_{j}'] = df['co2'].shift(-output_length-j)
        j = j + 1
        
    df = df.dropna(axis=0)
    
    return df

def get_chart_2(data):
    hover = alt.selection_single(
        fields=["episode"],
        nearest=True,
        on="mouseover",
        empty="none",
    )

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

    # Draw points on the line, and highlight based on selection
    points = lines.transform_filter(hover).mark_circle(size=60)

    # Draw a rule at the location of the selection
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