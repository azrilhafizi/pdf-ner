import streamlit as st
import spacy
from spacy_streamlit import visualize_ner

def ner(uploaded_file):
    content = uploaded_file.read().decode('utf-8')
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(content)

    visualize_ner(doc, labels=nlp.get_pipe("ner").labels)

def main():
    st.title('Case File NER')

    uploaded_file = st.file_uploader('Choose a text file', type=['txt'])

    if uploaded_file is not None:
        ner(uploaded_file)

if __name__ == '__main__':
    main()