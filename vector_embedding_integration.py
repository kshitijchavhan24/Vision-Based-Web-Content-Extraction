#!/usr/bin/env python3
import logging
import json
import os
import tempfile
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import CLIPTokenizer
from hdfs import InsecureClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the CLIP-compatible SentenceTransformer model.
model = SentenceTransformer('clip-ViT-B-32')

# Load the CLIP tokenizer with a maximum sequence length of 77 tokens.
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32', model_max_length=77)
MAX_SEQ_LENGTH = 77

def truncate_text(text):
    """
    Tokenizes the input text, truncates it to MAX_SEQ_LENGTH tokens,
    and decodes it back to text.
    """
    tokenized = tokenizer(text, truncation=False, add_special_tokens=True)
    tokens = tokenized['input_ids']
    if len(tokens) > MAX_SEQ_LENGTH:
        logger.info("Truncating text from %d tokens to %d tokens.", len(tokens), MAX_SEQ_LENGTH)
        tokens = tokens[:MAX_SEQ_LENGTH]
    truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)
    return truncated_text

def process_blocks(structured_data):
    """
    Processes each structured data block by truncating its text and generating an embedding.
    Expects each block to have a "block_text" key.
    
    :param structured_data: List of dicts with a "block_text" field.
    :return: List of numpy arrays representing embeddings.
    """
    embeddings = []
    for item in structured_data:
        # Use block_text if available, otherwise fallback to text, otherwise empty string.
        text_to_encode = item.get("block_text", item.get("text", ""))
        truncated_text = truncate_text(text_to_encode)
        try:
            text_embedding = model.encode(truncated_text, convert_to_numpy=True)
            embeddings.append(text_embedding)
            # Store the truncated text back as "block_text" to ensure consistency.
            item["block_text"] = truncated_text
        except Exception as e:
            logger.error("Error encoding text: %s", e)
    return embeddings

def list_hdfs_files(client, hdfs_path):
    """
    Recursively list all files (with full HDFS paths) under the given directory.
    """
    file_paths = []
    try:
        items = client.list(hdfs_path, status=True)
    except Exception as e:
        logger.error("Error listing %s: %s", hdfs_path, e)
        return file_paths

    for item, status in items:
        full_path = hdfs_path.rstrip('/') + '/' + item
        if status['type'] == 'DIRECTORY':
            file_paths.extend(list_hdfs_files(client, full_path))
        else:
            file_paths.append(full_path)
    return file_paths

def load_all_structured_data_from_hdfs(hdfs_client, hdfs_dir):
    """
    Loads all structured layout JSON files from the specified HDFS directory.
    Returns a merged list of all structured data blocks.
    """
    merged_data = []
    files = list_hdfs_files(hdfs_client, hdfs_dir)
    layout_files = [f for f in files if f.lower().endswith('_layout.json')]
    
    if not layout_files:
        logger.warning("No layout JSON files found in %s.", hdfs_dir)
        return merged_data

    for file in layout_files:
        try:
            with hdfs_client.read(file, encoding='utf-8') as reader:
                layout = json.load(reader)
            # For each block in this layout file, ensure it has a 'block_text' key.
            for block in layout:
                if "block_text" not in block:
                    if "text" in block:
                        block["block_text"] = block["text"]
                    else:
                        block["block_text"] = ""
                merged_data.append(block)
            logger.info("Loaded %d blocks from %s", len(layout), file)
        except Exception as e:
            logger.error("Error loading file %s: %s", file, e)
    return merged_data

def build_faiss_index(embeddings):
    """
    Builds a FAISS index using L2 (Euclidean) distance.
    """
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

def main():
    # Initialize HDFS client (adjust URL, user, and timeout as needed)
    hdfs_client = InsecureClient('http://localhost:9870', user='kc', timeout=300)
    
    # Specify the HDFS directory where the structured layout JSON files are stored.
    # This directory is created by your generate_layout_structure.py process.
    hdfs_structured_data_dir = '/structured_output'
    structured_data = load_all_structured_data_from_hdfs(hdfs_client, hdfs_structured_data_dir)
    
    if not structured_data:
        logger.warning("No structured data available for encoding.")
        return
    
    logger.info("Loaded a total of %d blocks of structured data.", len(structured_data))
    
    # Process the blocks: truncate text and generate embeddings.
    embeddings_list = process_blocks(structured_data)
    if not embeddings_list:
        logger.error("No embeddings generated.")
        return
    embeddings = np.vstack(embeddings_list)
    logger.info("Computed embeddings for %d blocks. Embeddings shape: %s",
                len(embeddings_list), embeddings.shape)
    
    # Build the FAISS index.
    index = build_faiss_index(embeddings)
    logger.info("FAISS index built with %d vectors.", index.ntotal)
    
    # Save the FAISS index to HDFS.
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        temp_index_path = tmp_file.name
    faiss.write_index(index, temp_index_path)
    hdfs_index_file = '/faiss/faiss_index.bin'
    try:
        hdfs_client.upload(hdfs_index_file, temp_index_path, overwrite=True)
        logger.info("FAISS index uploaded to HDFS at %s", hdfs_index_file)
    except Exception as e:
        logger.error("Error uploading FAISS index to HDFS: %s", e)
    finally:
        os.remove(temp_index_path)
    
    # Save the merged metadata back to HDFS.
    hdfs_metadata_output = '/faiss/merged_faiss_metadata.json'
    metadata_json = json.dumps(structured_data, indent=4)
    try:
        with hdfs_client.write(hdfs_metadata_output, encoding='utf-8', overwrite=True) as writer:
            writer.write(metadata_json)
        logger.info("Merged metadata saved to HDFS at %s", hdfs_metadata_output)
    except Exception as e:
        logger.error("Error writing metadata to HDFS: %s", e)
    
    # Sample query: run a test search against the FAISS index.
    query = "Submit button for the contact form"
    query_embedding = model.encode([query], convert_to_numpy=True)
    k = 3  # Retrieve the top 3 nearest neighbors.
    distances, indices = index.search(query_embedding, k)
    
    print("\nQuery:", query)
    print("Top {} results:".format(k))
    for i, idx in enumerate(indices[0]):
        print("\nResult {}:".format(i+1))
        # Now every block should have a "block_text" key.
        print("Text:", structured_data[idx]["block_text"])
        print("Distance:", distances[0][i])
    
if __name__ == "__main__":
    main()
