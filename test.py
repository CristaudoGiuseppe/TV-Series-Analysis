import altair as alt
import pandas as pd
from vega_datasets import data


def get_data():
    source = data.stocks()
    print(source)
    source = source[source.date.gt("2004-01-01")]
    return source

source = get_data()