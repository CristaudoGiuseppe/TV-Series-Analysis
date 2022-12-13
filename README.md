# TV Series Analysis
This is a Streamlit app that allows you to search for a TV series and analyze it. The app uses the **tv_series** and **utils** modules to search and retrieve information about the TV series.

To use this app, you need to have the following libraries installed:

* streamlit
* seaborn
* altair
* matplotlib
* pandas
* numpy

## How to use the app:

To start the app, run the following command:

```console
    streamlit run tv_series_analysis.py
```
This will start the app in your web browser. You can then search for a TV series by entering the name in the search bar and clicking the "Search" button. This will display the TV series information and allow you to analyze it.

## What the app does:

When you search for a TV series, the app will retrieve the following information about the TV series:

* Poster image
* Popularity
* Overview
* Number of seasons
* Number of episodes
* Episode run time
* Vote average
* Vote count
* In production status
* Watch providers and their logos
The app also calculates the total watch time needed to watch all episodes of the TV series.

In addition to this information, the app also provides a heatmap of the average votes for each season of the TV series, a bar chart of the total number of votes for each season, a pie chart of the percentage of the total watch time for each season, and a bar chart of the predicted popularity of the TV series in the future.

You can also download the full episode data for the TV series as a CSV file.

# tv_series.py

This python script uses the requests and json libraries to interact with the The Movie Database (TMDb) API to retrieve information about TV series. The script uses the pandas library to store and manipulate data, and the numpy and sklearn libraries to perform machine learning tasks.

## Dependencies:
* requests
* json
* pandas
* sklearn
* numpy

## Usage:

Before running the script, make sure to add your TMDb API key to the settings.json file in the following format:

```json
    {
        "API_KEY": "<YOUR_API_KEY>"
    }
```

To run the script, use the following command:

```console
    python tv_series.py
```

The script defines several functions that can be used to search for TV series by keyword, retrieve information about episode runtimes, get detailed information about a TV series, and perform machine learning tasks.

## Funcionts:

* **test_api_key** - tests whether the API key is valid by checking the API response for the specified movie
* **search_keywords** - searches for TV series by keyword and returns the most popular result
* **get_episode_runtime** - gets the information on how long episodes of a given TV series are
* **get_tv_series_general_information** - returns a dictionary with minor information about a TV series
* **get_tv_series_details** - returns a dictionary with detailed information about a TV series
* **get_tv_series_details_df** - returns a dataframe with detailed information about a TV series