from textblob import TextBlob
import pandas as pd
import streamlit as st
import cleantext

st.header('Sentimental Analysis')
with st.expander('Analyze Text'):
    text = st.text_input('Text here: ')
    if text:
        blob = TextBlob(text)
        st.write('Polarity: ', round(blob.sentiment.polarity,2))
        st.write('Subjectivity: ', round(blob.sentiment.subjectivity,2))

# NOW WE WILL CLEAN THE TEXT:
pre = st.text_input('CLean_Text: ')
if pre:
    st.write(cleantext.clean(pre, clean_all=False, extra_spaces=True, stemming=True,
    stopwords=True, lowercase=True, numbers=True, punct=True))

with st.expander('Analyze CSV'):
    upl = st.file_uploader('Upload file')
# DEFINING A SCORE FUNCTION THAT WE"RE GOING TO USE LATER IT"LL BASICALLY RETURN THE POLARITY OF THE SENTIMENT
    def score(x):
        blob1 = TextBlob(x)
        return blob1.sentiment.polarity
# SETTING BOUNDARIES FOR APPROPRIATE CLASSIFICATION
    def analyze(x):
        if x>=0.5:
            return 'POSITIVE'
        elif x<= -0.5:
            return 'NEGATIVE'
        else:
            return 'Neutral'

if upl:
    df = pd.read_excel(upl)
    del df['Unnamed: 0']
    df['score'] = df['tweets'].apply(score)
    df['analysis'] = df['score'].apply(analyze)
    st.write(df.head(10))

    @st.cache
    def convert_df(df):
        #IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    csv = convert_df(df)
# Creating the download button
    st.download_button(label="Download data as CSV",
    data=csv,
    file_name='sentimment.csv',
    mime='text/csv')