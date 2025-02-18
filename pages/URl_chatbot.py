import requests
from bs4 import BeautifulSoup
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import streamlit as st

# Load a sentence transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from a given URL
def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = ' '.join([p.text for p in soup.find_all('p')])
    return text

# Function to create FAISS index
def create_faiss_index(sentences):
    embeddings = model.encode(sentences)
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings, dtype=np.float32))
    return index, sentences

# Function to search in FAISS index
def search_faiss(query, index, sentences, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)
    results = [sentences[i] for i in indices[0]]
    return results

# Streamlit App UI
st.title("Website-based Chatbot")
url = st.text_input("Enter a website URL:")

if st.button("Crawl & Train"):
    with st.spinner("Extracting and processing data..."):
        text_data = extract_text_from_url(url)
        sentences = text_data.split('. ')
        index, stored_sentences = create_faiss_index(sentences)
        st.session_state['index'] = index
        st.session_state['stored_sentences'] = stored_sentences
        st.success("Crawling completed! You can now ask questions.")

if 'index' in st.session_state:
    user_query = st.text_input("Ask me anything based on the website:")
    if st.button("Get Answer"):
        response = search_faiss(user_query, st.session_state['index'], st.session_state['stored_sentences'])
        st.write("### Response:")
        for ans in response:
            st.write(f"- {ans}")
