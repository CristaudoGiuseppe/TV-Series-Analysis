import requests, json
import pandas as pd, utils, numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt


with open('settings.json') as settings:
  settings_json = json.load(settings)
  API_KEY = settings_json['API_KEY']
  
s = requests.session()

def test_api_key(api_key = API_KEY):
    r = (s.get(f'https://api.themoviedb.org/3/movie/76341?api_key={api_key}').text)
    #print(r_json)
    try:
        # wrong key error: {"status_code":7,"status_message":"Invalid API key: You must be granted a valid key.","success":false}
        if r['success'] == False:
            print(r['status_message'])
    except: # Everything Fine
        print('The API KEY is valid!')
    
# Search a TV Series by keywords and return the most popular result.
def search_keywords(keyword, api_key = API_KEY):
    r = json.loads(s.get(f'https://api.themoviedb.org/3/search/tv?query={keyword}&api_key={api_key}&page=1&include_adult=true').text)
    df = pd.DataFrame(columns=['ID', 'Name', 'Overview', 'Poster', 'Popularity'])
    
    # For each result in the response, populate the dataframe.
    for result in r['results']:
        df.loc[len(df.index)] = [result['id'], result['name'], result['overview'], f"https://image.tmdb.org/t/p/w440_and_h660_face/{result['poster_path']}", result['popularity']]
    
    # Drop NaN value, sort by descending popularity and reset DataFrame index
    df = df.dropna()
    df = df.sort_values(by=['Popularity'], ascending=False)
    df = df.reset_index()
    
    return df[0]

# Get the informaion on how long episodes are long given a TV Series ID
def get_episode_runtime(tv_series_id, api_key = API_KEY):
    r = json.loads(s.get(f'https://api.themoviedb.org/3/tv/{tv_series_id}?api_key={api_key}').text)
    
    # The information about episode runtime is encoded as [22] or [50, 60] 
    episode_rt = str(r['episode_run_time']).replace('[','').replace(']','') # [22] -> 22
    if ',' in episode_rt:
        episode_rt = episode_rt.split(',')[0] # 50, 60 -> 50
        
    return int(episode_rt)

# Return a Python Dictionary containing minor information about a TV Series
def get_tv_series_general_information(r, tv_series_id):
    result = {
        'vote_avg'              : r['vote_average'],
        'vote_count'            : r['vote_count'],
        'episode_rt'            : get_episode_runtime(tv_series_id),
        'in_production'         : r['in_production'],
        'number_of_episodes'    : r['number_of_episodes'],
        'number_of_seasons'     : r['number_of_seasons']
    }
    
    return result

# Get more information about a TV Series given a tv_series_id
def get_tv_series_overview(tv_series_id, api_key = API_KEY):
    r = json.loads(s.get(f'https://api.themoviedb.org/3/tv/{tv_series_id}?api_key={api_key}').text)
    minor_information_dict = get_tv_series_general_information(r, tv_series_id)
    
    df = pd.DataFrame(columns=['ID', 'Name', 'Air Date', 'Season Number', 'Poster', 'Episode Count'])
    
    for season in r['seasons']:
        df.loc[len(df.index)] = [season['id'], season['name'], season['air_date'], season['season_number'], f"https://image.tmdb.org/t/p/w440_and_h660_face/{season['poster_path']}" , season['episode_count']]
    
    return minor_information_dict, df

# Get watch providers information about a TV Series given a tv_series_id
def get_tv_series_watch_providers(tv_series_id, api_key = API_KEY):
    r = json.loads(s.get(f'https://api.themoviedb.org/3/tv/{tv_series_id}/watch/providers?api_key={api_key}').text)
    
    df = pd.DataFrame(columns=['Country', 'Provider Name', 'Provider Logo'])
    
    for res in r['results']:
        # {'link': 'https://www.themoviedb.org/tv/456-the-simpsons/watch?locale=ZA', 'flatrate': [{'logo_path': '/7rwgEs15tFwyR9NPQ5vpzxTj19Q.jpg', 'provider_id': 337, 'provider_name': 'Disney Plus', 'display_priority': 28}]}
        df.loc[len(df.index)] = [res, r['results'][res]['flatrate'][0]['provider_name'], f"https://image.tmdb.org/t/p/w440_and_h660_face/{r['results'][res]['flatrate'][0]['logo_path']}"]
    
    return df

def get_heatmap_avg_vote(seasons):
    df = pd.DataFrame()
    df.index.name = 'Episode'
    i = 1
    for s in seasons:
        df[f'Season {i}'] = s['Vote AVG.']
        i += 1
    
    return(df)

def get_heatmap_vote_count(seasons):
    df = pd.DataFrame()
    df.index.name = 'Episode'
    i = 1
    for s in seasons:
        df[f'Season {i}'] = s['Vote Count']
        i += 1
    
    return(df)
    

def get_series_data(tv_show_id, api_key = API_KEY):
    seasons = []
    episodes_total_time = []
    votes = []
    episode_rt = get_episode_runtime(tv_show_id)
    episodes_total_time.append(episode_rt)
    
    r = s.get(f'https://api.themoviedb.org/3/tv/{tv_show_id}?api_key={api_key}')
    r_json = json.loads(r.text)    
    
    df_avg_chart = pd.DataFrame(columns=['season', 'episode', 'avg', 'count'])
    
    #df_total_time = pd.DataFrame(columns=['Total Time'])
    #df_total_time.index.name = 'Episode'
    #df_total_time['Total Time'][0] = episode_rt
    
    for season in r_json['seasons']:
        if season['name'] == 'Specials':
            continue
        temp = get_season_overview(tv_show_id, season['season_number'])
        
        for index, row in temp.iterrows():
            df_avg_chart.loc[len(df_avg_chart.index)] = [season['name'], index + 1, row['Vote AVG.'], row['Vote Count']]
            episodes_total_time.append(episodes_total_time[-1] + episode_rt)
            votes.append(row['Vote AVG.'])
        seasons.append(temp)
    
    df_total_time = pd.DataFrame({'time': episodes_total_time, 'episodes':range(0, len(episodes_total_time))})
    df_total_time.index.name = 'Episode'
    
    df_total_vote = pd.DataFrame({'Vote': votes})
    df_total_vote.index.name = 'Episode'
    df_total_vote_copy = df_total_vote.copy()
    df_total_vote_copy['y'] = df_total_vote['Vote'].shift(-1)
    train = df_total_vote_copy[:-int(len(df_total_vote_copy)/2)]
    test = df_total_vote_copy[-int(len(df_total_vote_copy)/2):]
    test = test.drop(test.tail(1).index) # Drop last row
    test = test.copy()
    test['baseline_pred'] = test['Vote']
    
    X_train = train['Vote'].values.reshape(-1,1)
    y_train = train['y'].values.reshape(-1,1)
    X_test = test['Vote'].values.reshape(-1,1)
    # Initialize the model
    dt_reg = DecisionTreeRegressor(random_state=42)
    # Fit the model
    dt_reg.fit(X=X_train, y=y_train)
    # Make predictions
    dt_pred = dt_reg.predict(X_test)
    # Assign predictions to a new column in test
    test['dt_pred'] = dt_pred
    gbr = GradientBoostingRegressor(random_state=42)
    gbr.fit(X_train, y=y_train.ravel())
    gbr_pred = gbr.predict(X_test)
    test['gbr_pred'] = gbr_pred
    
    def mape(y_true, y_pred):
        return round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 2)
    
    
    baseline_mape = mape(test['y'], test['baseline_pred'])
    dt_mape = mape(test['y'], test['dt_pred'])
    gbr_mape = mape(test['Vote'], test['gbr_pred'])
    # Generate bar plot
    fig, ax = plt.subplots(figsize=(7, 5))
    x = ['Baseline', 'Decision Tree', 'Gradient Boosting']
    y = [baseline_mape, dt_mape, gbr_mape]
    ax.bar(x, y, width=0.4)
    ax.set_xlabel('Regressor models')
    ax.set_ylabel('MAPE (%)')
    ax.set_ylim(0, 0.3)
    for index, value in enumerate(y):
        plt.text(x=index, y=value + 0.02, s=str(value), ha='center')
    
    plt.tight_layout()
    test['episodes'] = range(0, len(test))
    #print(test)
    df_prediction = pd.DataFrame(columns=['Type', 'Vote', 'episode'])
    for index, row in test.iterrows():
        df_prediction.loc[len(df_prediction.index)] = ['Vote', row['Vote'], index]
        df_prediction.loc[len(df_prediction.index)] = ['dt_pred', row['dt_pred'], index]
        df_prediction.loc[len(df_prediction.index)] = ['gbr_pred', row['gbr_pred'], index]
    for index, row in train.iterrows():
        df_prediction.loc[len(df_prediction.index)] = ['Vote', row['Vote'], index]
    #print(df_prediction)    
    
    df_heatmap_vote = get_heatmap_avg_vote(seasons)
    df_heatmap_vote_count = get_heatmap_vote_count(seasons)
    return df_heatmap_vote, df_heatmap_vote_count, df_avg_chart, df_total_time, df_prediction

def get_season_overview(tv_show_id, season_number, api_key = API_KEY):
    r = s.get(f'https://api.themoviedb.org/3/tv/{tv_show_id}/season/{season_number}?api_key={api_key}')
    r_json = json.loads(r.text)
    df = pd.DataFrame(columns=['ID', 'Name', 'Air Date', 'Vote AVG.', 'Vote Count'])
    for episode in r_json['episodes']:
        df.loc[len(df.index)] = [episode['id'], episode['name'], episode['air_date'], episode['vote_average'], episode['vote_count']]
    return df
        
    
#search_keywords(API_KEY, 'sex education')
# get_tv_series_overview(API_KEY, 456)
# get_season_overview(API_KEY, 456, 2)
#get_series_data(81356)
#test_api_key()
get_tv_series_watch_providers(456)
    