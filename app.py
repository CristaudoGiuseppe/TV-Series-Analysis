import streamlit as st 
import tv_series, utils
import seaborn as sns, altair as alt
import matplotlib.pyplot as plt
import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title = "TV Series Analysis",
    page_icon = "tv"
)

def analyze_tv_series(search):
    df = tv_series.search_keywords(keyword = search)
    vote_avg, vote_count, episode_rt, in_production, number_of_episodes, number_of_seasons, df_id = tv_series.get_tv_series_overview(id = df['ID'][0])
    st.header(df['Name'][0])
    c1, c2 = st.columns(2)
    with c1:
        st.image(df['Poster'][0])
        st.caption(f"Popularity: {df['Popularity'][0]}")
        st.caption(f"Number of season(s): {number_of_seasons}")
        st.caption(f"Number of episodes: {number_of_episodes}")
        st.caption(f"Episode run time: {episode_rt}")
    with c2:
        st.subheader('Overview')
        st.markdown(df['Overview'][0])
        
        with st.expander("Show more info"):
            st.caption(f"Vote avarage: {vote_avg}")
            st.caption(f"Vote count: {vote_count}")
            st.caption(f"In production?: {in_production}") # magare fare verde se è in produzione rosso in caso contrario
            # inserire info su watch provider
    
    minutes_needed = number_of_seasons * number_of_episodes * episode_rt
    st.caption(f'Total watch time needed: {minutes_needed} minutes or {minutes_needed/60} hours')
    # formattare correttamente
    
    st.subheader(f"{df['Name'][0]} overview")
    st.dataframe(df_id)
    # show more information con tutti gli episodi
    
    df_avg_vote, df_vote_count, df_avg_chart, df_total_time, df_prediction = tv_series.get_series_data(df['ID'][0])
    c1, c2 = st.columns(2)
    with c1:
        st.subheader('Avarage Vote Heatmap')
        fig, ax = plt.subplots()
        sns.color_palette("YlOrBr", as_cmap=True)
        sns.heatmap(df_avg_vote, ax=ax, annot = True, fmt=".1f", linewidth=.5)
        st.write(fig)
    with c2:
        st.subheader('Vote Count Heatmap')
        fig, ax = plt.subplots()
        sns.color_palette("YlOrBr", as_cmap=True)
        sns.heatmap(df_vote_count, ax=ax, annot = True, linewidth=.5)
        st.write(fig)
    
    all_seasons = df_avg_chart.season.unique()
    seasons = st.multiselect("Choose season to visualize", all_seasons, all_seasons[:3])
    
    # space(1)
    df_avg_chart = df_avg_chart[df_avg_chart.season.isin(seasons)]
    chart = utils.get_chart(df_avg_chart)
    st.altair_chart(chart, use_container_width=True)
    
    sel_range = st.slider("Episode Interval", value=(0, len(df_total_time)), min_value = 0, max_value = len(df_total_time-1))
    time_chart = alt.Chart(df_total_time[sel_range[0]:sel_range[1]], title="Total Watch Time in Minutes").mark_line().encode(
        x = "episodes",
        y = "time"
    )
    st.altair_chart(time_chart, use_container_width=True)
    st.caption(f"Total time to watch: {df_total_time['time'][sel_range[1] - 1] - df_total_time['time'][sel_range[0]]} minutes")
    
    all_prediction = df_prediction.Type.unique()
    predictions = st.multiselect("Choose prediction to visualize", all_prediction, all_prediction[:3])
    
    # space(1)
    df_prediction = df_prediction[df_prediction.Type.isin(predictions)]
    prediction_chart = utils.get_chart_2(df_prediction)
    st.altair_chart(prediction_chart, use_container_width=True)

   

    
    


st.title(":tv: TV Series Analysis")
# magare inserire come funziona
search = st.text_input("Input a TV Series")

if 'search' not in st.session_state:
    st.session_state.search = search
if 'analyze' not in st.session_state:
    st.session_state.analyze = False

analyze = st.button("ANALYZE")
if analyze:
    st.session_state.analyze = True 
    analyze_tv_series(search)
elif st.session_state.analyze:
    analyze_tv_series(search)
    

# IDEE INFORMAZIONI
# correlazione voti medi e numero di voti
# predict next episode vote based on previous seasons
# download csv con info sulle informazioni sulla serie


# sistemare errore in heatmap numero di episodi errato o troppi episodi vedi simpson o big bang theory

# refactoring
# spiegare codice
# spiegare nell'app
# fare bene grafico heatmap
# sistemare errori nn tutti gli episodi