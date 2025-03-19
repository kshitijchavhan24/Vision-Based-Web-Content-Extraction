#!/usr/bin/env python3
import os
import json
import tempfile
import logging
import numpy as np
import faiss
import streamlit as st

from hdfs import InsecureClient
from sentence_transformers import SentenceTransformer
from transformers import CLIPTokenizer, pipeline

# ------------------------------------------------------------------------------
# NOTE:
# If you encounter errors related to torch's __path__ (e.g., during file watching),
# try running Streamlit with auto-reload disabled:
#    streamlit run streamlit_app.py --server.runOnSave false
# ------------------------------------------------------------------------------

# ----------------------------
# Configuration & Logging
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HDFS Settings
HDFS_URL = "http://localhost:9870"
HDFS_USER = "kc"
INDEX_HDFS_PATH = "/faiss/faiss_index.bin"
METADATA_HDFS_PATH = "/faiss/merged_faiss_metadata.json"

# Similarity threshold for cosine similarity (using inner product)
# (For cosine similarity, values closer to 1.0 indicate a perfect match.)
SIMILARITY_THRESHOLD = 0.85  # Adjust as needed

# Initialize HDFS client
hdfs_client = InsecureClient(HDFS_URL, user=HDFS_USER, timeout=300)

# ----------------------------
# Functions to Load Artifacts from HDFS
# ----------------------------
def load_faiss_index_from_hdfs():
    """Downloads the FAISS index from HDFS to a temporary file and loads it."""
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
    """Loads the merged metadata JSON from HDFS."""
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
@st.cache_resource
def load_models():
    st.write("Loading models, please wait...")
    # Retrieval model for encoding queries
    retrieval_model = SentenceTransformer('clip-ViT-B-32')
    # Tokenizer for truncating text if needed
    clip_tokenizer = CLIPTokenizer.from_pretrained(
        'openai/clip-vit-base-patch32', model_max_length=77)
    # Instruction-tuned generative model for detailed answer generation.
    # (Generation parameters are set later in generate_full_response.)
    gen_pipeline = pipeline("text-generation", model="gpt2")
    return retrieval_model, clip_tokenizer, gen_pipeline

retrieval_model, clip_tokenizer, gen_pipeline = load_models()

MAX_SEQ_LENGTH = 77

def truncate_text(text):
    """Tokenizes the text, truncates it to MAX_SEQ_LENGTH tokens, and decodes it back."""
    tokenized = clip_tokenizer(text, truncation=False, add_special_tokens=True)
    tokens = tokenized['input_ids']
    if len(tokens) > MAX_SEQ_LENGTH:
        logger.info("Truncating text from %d tokens to %d tokens.", len(tokens), MAX_SEQ_LENGTH)
        tokens = tokens[:MAX_SEQ_LENGTH]
    return clip_tokenizer.decode(tokens, skip_special_tokens=True)

def get_query_embedding(query):
    """
    Encodes the query using the retrieval model and normalizes the embedding for cosine similarity.
    """
    emb = retrieval_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(emb)
    return emb

def build_faiss_index(embeddings):
    """
    Normalizes embeddings and builds a FAISS index using inner product (for cosine similarity).
    """
    faiss.normalize_L2(embeddings)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index

def retrieve_context(query, index, metadata, k=3):
    """
    Searches the FAISS index for the top k blocks similar to the query.
    If the highest similarity score is below the threshold, returns None.
    Otherwise, extracts the text from each block (using "block_text" if available,
    fallback to "text"), deduplicates them (preserving order), and joins them with newlines.
    """
    query_emb = get_query_embedding(query)
    distances, indices = index.search(query_emb, k)
    # If the best match is below the threshold, consider no relevant data available.
    if distances[0][0] < SIMILARITY_THRESHOLD:
        return None, distances, indices
    retrieved_blocks = [metadata[i] for i in indices[0] if i < len(metadata)]
    context_blocks = [block.get("block_text", block.get("text", "")) for block in retrieved_blocks]
    # Deduplicate while preserving order
    unique_contexts = list(dict.fromkeys(context_blocks))
    context = "\n".join(unique_contexts)
    return context, distances, indices

def generate_full_response(query, retrieved_context):
    """
    Generates a full, ChatGPT-like response using a pretrained generative model,
    based solely on the retrieved context and the original query.
    """
    prompt = (
        f"Using only the information below, provide a detailed and clear answer in a natural, conversational style:\n\n"
        f"Information:\n{retrieved_context}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )
    # Adjusted generation settings to help avoid repetitive output.
    result = gen_pipeline(
        prompt,
        max_length=250,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )
    return result[0]['generated_text']

@st.cache_resource
def load_faiss_and_metadata():
    """Loads the FAISS index and metadata from HDFS."""
    index = load_faiss_index_from_hdfs()
    metadata = load_metadata_from_hdfs()
    return index, metadata

index, metadata = load_faiss_and_metadata()

try:
    import torch
    if hasattr(torch, "_classes"):
        class ListWithPath(list):
            @property
            def _path(self):
                return self
        torch._classes.__path__ = ListWithPath([])
except Exception as e:
    # In case anything goes wrong with the monkey patch, log or ignore.
    sys.stderr.write(f"Warning: Failed to patch torch._classes.__path__: {e}\n")

    

# ----------------------------
# Streamlit User Interface
# ----------------------------
st.title("Scraped Data Q&A System")
st.write("Ask a question related to the scraped content from your URLs.")

user_query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if not user_query:
        st.write("Please enter a question.")
    else:
        if index is None or not metadata:
            st.write("Error: Retrieval system not properly loaded.")
        else:
            with st.spinner("Processing your question..."):
                retrieved_context, distances, indices = retrieve_context(user_query, index, metadata, k=3)
            # If no relevant context is found, show a fallback message.
            if retrieved_context is None or retrieved_context.strip() == "":
                st.markdown("### No Relevant Data Found")
                st.write("No relevant data found. Please ask a question related to the scraped URLs.")
            else:
                # Section 1: Display the raw retrieved context.
                st.markdown("### Answer (Retrieved Context)")
                st.write(retrieved_context)
                
                # Section 2: Generate a full response using the generative model.
                try:
                    full_response = generate_full_response(user_query, retrieved_context)
                except Exception as e:
                    logger.error("Error during response generation: %s", e)
                    full_response = "An error occurred while generating the response."
                st.markdown("### Response (Full Answer)")
                st.write(full_response)
                
                # Section 3: Display similarity scores.
                st.markdown("### Similarity Scores (Cosine Similarity)")
                st.write(distances)
