#!/usr/bin/env python3
import os
import json
import tempfile
import logging
import numpy as np
import streamlit as st
import faiss

from hdfs import InsecureClient
from sentence_transformers import SentenceTransformer
from transformers import CLIPTokenizer, pipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# HDFS and File Paths Settings
# ----------------------------
HDFS_URL = "http://localhost:9870"
HDFS_USER = "kc"
INDEX_HDFS_PATH = "/faiss/faiss_index.bin"
METADATA_HDFS_PATH = "/faiss/merged_faiss_metadata.json"

# Similarity threshold for retrieval (L2 distance; lower is better)
SIMILARITY_THRESHOLD = 1.0  # adjust this value as needed

# Initialize HDFS client
hdfs_client = InsecureClient(HDFS_URL, user=HDFS_USER, timeout=300)

# ----------------------------
# Functions to Load Artifacts from HDFS
# ----------------------------
def load_faiss_index_from_hdfs():
    """
    Downloads the FAISS index from HDFS to a temporary file and reads it.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            temp_index_path = tmp_file.name
        hdfs_client.download(INDEX_HDFS_PATH, temp_index_path, overwrite=True)
        index = faiss.read_index(temp_index_path)
        os.remove(temp_index_path)
        logger.info("Loaded FAISS index from HDFS.")
        return index
    except Exception as e:
        logger.error("Error loading FAISS index from HDFS: %s", e)
        return None

def load_metadata_from_hdfs():
    """
    Reads the merged metadata JSON from HDFS.
    """
    try:
        with hdfs_client.read(METADATA_HDFS_PATH, encoding='utf-8') as reader:
            metadata = json.load(reader)
        logger.info("Loaded metadata from HDFS.")
        return metadata
    except Exception as e:
        logger.error("Error loading metadata from HDFS: %s", e)
        return []

# ----------------------------
# Load Models (Retrieval & Generation)
# ----------------------------
@st.cache(allow_output_mutation=True)
def load_models():
    st.write("Loading models...")
    # Retrieval model for generating embeddings
    retrieval_model = SentenceTransformer('clip-ViT-B-32')
    clip_tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32', model_max_length=77)
    # Generative model pipeline (for generating natural language responses)
    gen_pipeline = pipeline("text2text-generation", model="t5-small")
    return retrieval_model, clip_tokenizer, gen_pipeline

retrieval_model, clip_tokenizer, gen_pipeline = load_models()

# ----------------------------
# Utility Functions
# ----------------------------
MAX_SEQ_LENGTH = 77

def truncate_text(text):
    """
    Truncates text to a maximum of MAX_SEQ_LENGTH tokens using the CLIP tokenizer.
    """
    tokenized = clip_tokenizer(text, truncation=False, add_special_tokens=True)
    tokens = tokenized['input_ids']
    if len(tokens) > MAX_SEQ_LENGTH:
        logger.info("Truncating text from %d tokens to %d tokens.", len(tokens), MAX_SEQ_LENGTH)
        tokens = tokens[:MAX_SEQ_LENGTH]
    truncated_text = clip_tokenizer.decode(tokens, skip_special_tokens=True)
    return truncated_text

def get_query_embedding(query):
    """
    Encodes the user query into an embedding using the retrieval model.
    """
    return retrieval_model.encode([query], convert_to_numpy=True)

def retrieve_context(query, index, metadata, k=3):
    """
    Retrieves the top k similar blocks from the FAISS index.
    Returns the concatenated context text if the top result's distance is below threshold;
    otherwise, returns None.
    """
    query_emb = get_query_embedding(query)
    distances, indices = index.search(query_emb, k)
    # Check if best (lowest) distance is acceptable
    if distances[0][0] > SIMILARITY_THRESHOLD:
        return None, distances, indices
    # Retrieve corresponding blocks from metadata
    retrieved_blocks = [metadata[i] for i in indices[0] if i < len(metadata)]
    # Each block is expected to have a "block_text" field (ensured during embedding integration)
    context = " ".join(block.get("block_text", "") for block in retrieved_blocks)
    return context, distances, indices

def generate_answer(query, context):
    """
    Uses the generative model to produce an answer based on the context and query.
    """
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    result = gen_pipeline(prompt, max_length=150, do_sample=False)
    return result[0]['generated_text']

@st.cache(allow_output_mutation=True)
def load_faiss_and_metadata():
    """
    Loads the FAISS index and metadata from HDFS.
    """
    index = load_faiss_index_from_hdfs()
    metadata = load_metadata_from_hdfs()
    return index, metadata

# Load FAISS index and metadata (cached)
index, metadata = load_faiss_and_metadata()

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Scraped Data Q&A System")
st.write("Ask a question related to the scraped URLs.")

user_query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if not user_query:
        st.write("Please enter a question.")
    else:
        if index is None or not metadata:
            st.write("Error: Retrieval system not properly loaded.")
        else:
            context, distances, indices = retrieve_context(user_query, index, metadata, k=3)
            if context is None or context.strip() == "":
                st.write("No relevant data found. Please ask a question related to the scraped URLs.")
            else:
                answer = generate_answer(user_query, context)
                st.write("**Answer:**", answer)
                st.write("**Retrieved Context:**", context)
                st.write("**Distances:**", distances)
