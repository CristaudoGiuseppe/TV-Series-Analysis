import streamlit as st 
import tv_series, utils
import seaborn as sns, altair as alt
import matplotlib.pyplot as plt
import pandas as pd, numpy as np

st.set_page_config(
    page_title = "TV Series Analysis",
    page_icon = "tv"
)

def analyze_tv_series(search):
    df = tv_series.search_keywords(keyword = search)
    
    minor_information_dict, df_id = tv_series.get_tv_series_overview(df['ID'][0])
    st.header(df['Name'][0])
    c1, c2 = st.columns(2)
    with c1:
        st.image(df['Poster'][0])
        st.caption(f"Popularity: {df['Popularity'][0]}")
        st.caption(f"Number of season(s): {minor_information_dict.get('number_of_seasons')}")
        st.caption(f"Number of episodes: {minor_information_dict.get('number_of_episodes')}")
        st.caption(f"Episode run time: {minor_information_dict.get('episode_rt')}")
    with c2:
        st.subheader('Overview')
        st.markdown(df['Overview'][0])
        
        with st.expander("Show more info"):
            st.caption(f"Vote avarage: {minor_information_dict.get('vote_avg')}")
            st.caption(f"Vote count: {minor_information_dict.get('vote_count')}")
            if minor_information_dict.get('in_production'):
                ip = f'<p style="font-family:sans-serif; color:Green; font-size: 15px;">IN PRODUCTION</p>'
            else:
                 ip = f'<p style="font-family:sans-serif; color:Red; font-size: 15px;">IN PRODUCTION</p>'
            st.markdown(ip, unsafe_allow_html=True)
            
            df_watch_provider = tv_series.get_tv_series_watch_providers(df['ID'][0])
            # Create a list of options from the dataframe column
            options = df_watch_provider["Country"].tolist()
            # Create the selectbox widget
            selected_option = st.selectbox("Choose a country", options)
            selected_row = df_watch_provider.loc[df_watch_provider['Country'] == selected_option].values
            st.caption(f"Provider name: {selected_row[0][1]}")
            st.image(selected_row[0][2], width = 100)

                        
    
    minutes_needed = minor_information_dict.get('number_of_seasons') * minor_information_dict.get('number_of_episodes') * minor_information_dict.get('episode_rt')
    st.caption(f'Total watch time needed: {minutes_needed} minutes or {minutes_needed/60:.2f} hours')
        
    df_avg_vote, df_vote_count, df_avg_chart, df_total_time, df_prediction, df_full_info = tv_series.get_series_data(df['ID'][0])

    st.subheader(f"{df['Name'][0]} overview")
    st.dataframe(df_id)
        
    st.download_button('Downalod full episodes data', df_full_info.to_csv(), 'text/csv')
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader('Avarage Vote Heatmap')
        fig, ax = plt.subplots()
        sns.color_palette("YlOrBr", as_cmap=True)
        sns.heatmap(df_avg_vote, ax=ax, annot = True, fmt=".1f", linewidth=.5)
        
        # Find the dimension of a cell in order to get the right size of the font
        x_cell_size = ax.get_xaxis().get_tick_padding()
        y_cell_size = ax.get_yaxis().get_tick_padding()
        font_size = min(x_cell_size, y_cell_size)
        fig, ax = plt.subplots()
        sns.heatmap(df_avg_vote, ax=ax, annot = True, fmt=".1f", linewidth=.5, annot_kws={'size': font_size})
        st.write(fig)
    with c2:
        st.subheader('Vote Count Heatmap')
        fig, ax = plt.subplots()
        sns.color_palette("YlOrBr", as_cmap=True)
        sns.heatmap(df_vote_count, ax=ax, annot = True, linewidth=.5, annot_kws={'size': font_size})
        st.write(fig)
    
    st.subheader('Correlation between vote avg. and vote count')
    df_corr = df_avg_vote.corrwith(df_vote_count)
    df_corr = pd.DataFrame({'Correlation Coefficient' : df_corr.values})
    st.dataframe(df_corr)
    
    # Retrieves an array of unique season values from the df_avg_chart DataFrame.  
    all_seasons = df_avg_chart.season.unique() 
    # The user can select one or more seasons to visualize.
    seasons = st.multiselect("Choose season to visualize", all_seasons, all_seasons[:3])
    # Filters the df_avg_chart DataFrame to include only the rows with seasons that were selected by the user.
    df_avg_chart = df_avg_chart[df_avg_chart.season.isin(seasons)]
    chart = utils.get_chart(df_avg_chart, 0)
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
    with st.expander("How prediction works"):
        markdown = "### Decision Tree and Gradient Boosting for Time Series Prediction\n A decision tree is a supervised learning algorithm that can be used for both classification and regression tasks, including time series prediction. The algorithm trains on historical data to learn the relationship between the time series and the target variable, and uses this relationship to make predictions on new data.\n\n Gradient boosting is an ensemble learning method that combines the predictions of multiple individual models, such as decision trees, to create a more accurate and robust model. To make predictions on a time series using gradient boosting, the algorithm trains multiple decision trees on the historical data. Each tree makes predictions on the time series, and the predicted values are combined to create a final prediction. This process is repeated until the desired level of accuracy is reached."
        st.markdown(markdown)
        st.markdown("### How it works")
        f1 = "1. Format the dataset so that the current observation is a feature to predict the next observation (the target). This can be done by adding a second column that shifts the original data column such that the value in the first time step is now a predictor for the value in the second time step."
        st.markdown(f1)
        f2 = "2. Separate the dataset into a training and test set, using the last two years of data for the training set."
        st.markdown(f2)
        f3 = "3. Create a baseline model that naively predicts that the next observation will have the same value as the current observation."
        st.markdown(f3)
        f4 = "4. Apply a decision tree regressor to the training data and make predictions on the test set."
        st.markdown(f4)
        f5 = "5. Apply gradient boosting to the training data and make predictions on the test set."
        st.markdown(f5)


    
    df_prediction = df_prediction[df_prediction.Type.isin(predictions)]
    prediction_chart = utils.get_chart(df_prediction, 1)
    st.altair_chart(prediction_chart, use_container_width=True)


st.title(":tv: TV Series Analysis")
search = st.text_input("Input a TV Series")

if 'search' not in st.session_state:
    st.session_state.search = search

if search:
    analyze_tv_series(search)
    
    





