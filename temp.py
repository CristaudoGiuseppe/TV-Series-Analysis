import streamlit as st 
import tv_series


# st.title / st.header / st.subheader / st.write
# st.title("", anchor="page-name")
# st.markdown # permette di inserire formule matematiche col formato latex
# st.caption # st.text
# st.code # st.latex

# TABLE st.table / st.dataframe lo mostra tutto, accetta panda stiler per colorare 
# st.json
#st.altair_chart --> grafici interattivi
# st.line/bar/area_chart non customizzabili -> per farlo usare altair
# st.plotly / st.blokeh per grafici interattivi
# st.image(url)
# INTERACTIVE LIBRARY
# primo campo è il nome, disable=True per disabilitare, key="" widget identity, on_click e assegni quello da seguire
# clicked = st.button("Click me")
# st.checkbox, st.radio -> potrebbe essere usato per selezionare la serie
# st.selectbox -> creare un metodo per ogni opzione e mapparlo
# search = st.text_input("")
# if search == "": st.stop()
# st.file_uploader
# st.download_button
# c1, c2, c3 = st.columns(3)
# with c1: st.head()
# with st.spinner("Uploading"): upload_data
#st.sidebar

def analyze_tv_series(index):
    st.header(index)
    pass

def click_search_button(search):
    df = tv_series.search_keywords(keyword = search)
    c1, c2 = st.columns(2)
    with c1:
        for i in range(0, 2):
            st.subheader(df['Name'][i])
            st.image(df['Poster'][i], width = 200)
            st.caption(f"Popularity: {df['Popularity'][i]}")
    with c2:
        for i in range(2, 4):
            st.subheader(df['Name'][i])
            st.image(df['Poster'][i], width = 200)
            st.caption(f"Popularity: {df['Popularity'][i]}")
            
        
    select_tv_series = st.selectbox("Choose TV Series", 
                                    ("None", df['Name'][0], df['Name'][1], df['Name'][2], df['Name'][3]), 
                                    key = 'select_tv_series')
  
    
    
st.set_page_config(
    page_title = "TV Series Analysis",
    page_icon = "tv"
)

# settare bene guardando il video

st.title(":tv: TV Series Analysis")
search = st.text_input("Input a TV Series")

if st.button("SEARCH"):
    click_search_button(search)


    
    
