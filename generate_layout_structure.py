# generate_layout_structure.py

import json
import os
from hdfs import InsecureClient

def group_words_into_lines(words, vertical_threshold=10):
    """
    Group words into lines based on their 'top' coordinate.
    Words with a 'top' difference less than vertical_threshold are considered on the same line.
    """
    words = sorted(words, key=lambda x: x['top'])
    lines = []
    current_line = []
    current_top = None
    for word in words:
        if not current_line:
            current_line.append(word)
            current_top = word['top']
        else:
            if abs(word['top'] - current_top) < vertical_threshold:
                current_line.append(word)
            else:
                current_line = sorted(current_line, key=lambda x: x['left'])
                lines.append(current_line)
                current_line = [word]
                current_top = word['top']
    if current_line:
        current_line = sorted(current_line, key=lambda x: x['left'])
        lines.append(current_line)
    return lines

def group_lines_into_blocks(lines, block_vertical_threshold=20):
    """
    Group lines into blocks.
    Lines that are vertically close (gap less than block_vertical_threshold) are grouped into one block.
    """
    blocks = []
    current_block = []
    for line in lines:
        line_top = line[0]['top']
        if not current_block:
            current_block.append(line)
        else:
            last_line = current_block[-1]
            last_line_bottom = max(word['top'] + word['height'] for word in last_line)
            if line_top - last_line_bottom < block_vertical_threshold:
                current_block.append(line)
            else:
                blocks.append(current_block)
                current_block = [line]
    if current_block:
        blocks.append(current_block)
    return blocks

def compute_bounding_box_for_line(line):
    """Compute the bounding box for a line of words."""
    left = min(word['left'] for word in line)
    top = min(word['top'] for word in line)
    right = max(word['left'] + word['width'] for word in line)
    bottom = max(word['top'] + word['height'] for word in line)
    return [left, top, right, bottom]

def compute_bounding_box_for_block(block):
    """Compute the bounding box for a block of lines."""
    boxes = [compute_bounding_box_for_line(line) for line in block]
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[2] for box in boxes)
    bottom = max(box[3] for box in boxes)
    return [left, top, right, bottom]

def merge_line_text(line):
    """Merge the texts of words in a line into a single string."""
    return " ".join(word['text'] for word in line)

def generate_layout_structure(ocr_data):
    """
    Given OCR data (a list of word detections), generate a structured representation.
    The structure groups words into lines, then groups lines into blocks.
    """
    lines = group_words_into_lines(ocr_data, vertical_threshold=10)
    blocks = group_lines_into_blocks(lines, block_vertical_threshold=20)
    
    structured_blocks = []
    for block in blocks:
        block_text = "\n".join(merge_line_text(line) for line in block)
        block_box = compute_bounding_box_for_block(block)
        lines_data = []
        for line in block:
            line_text = merge_line_text(line)
            line_box = compute_bounding_box_for_line(line)
            lines_data.append({
                "text": line_text,
                "bounding_box": line_box
            })
        structured_blocks.append({
            "text": block_text,
            "bounding_box": block_box,
            "lines": lines_data
        })
    return structured_blocks

def list_hdfs_files(client, hdfs_path):
    """
    Recursively list all files in the given HDFS directory.
    Returns a list of full HDFS file paths.
    """
    file_paths = []
    try:
        items = client.list(hdfs_path, status=True)
    except Exception as e:
        print(f"Error listing {hdfs_path}: {e}")
        return file_paths

    for item, status in items:
        full_path = hdfs_path.rstrip('/') + '/' + item
        if status['type'] == 'DIRECTORY':
            file_paths.extend(list_hdfs_files(client, full_path))
        else:
            file_paths.append(full_path)
    return file_paths

def main():
    # Initialize HDFS client (adjust URL, user, and timeout as needed)
    hdfs_client = InsecureClient('http://localhost:9870', user='kc', timeout=300)
    
    # Define HDFS directories for OCR input and structured layout output
    input_hdfs_dir = '/ocr_output'
    output_hdfs_dir = '/structured_output'
    
    # Ensure the output directory exists on HDFS
    try:
        hdfs_client.makedirs(output_hdfs_dir)
    except Exception as e:
        print(f"Error creating HDFS output directory {output_hdfs_dir}: {e}")
    
    # Recursively list all OCR JSON files in the input directory
    all_files = list_hdfs_files(hdfs_client, input_hdfs_dir)
    ocr_files = [f for f in all_files if f.lower().endswith('_ocr.json')]
    
    if not ocr_files:
        print("No OCR JSON files found in HDFS directory:", input_hdfs_dir)
        return
    
    for ocr_file in ocr_files:
        print(f"Processing OCR file: {ocr_file}")
        try:
            with hdfs_client.read(ocr_file, encoding='utf-8') as reader:
                ocr_data = json.load(reader)
        except Exception as e:
            print(f"Error reading OCR file {ocr_file}: {e}")
            continue
        
        layout = generate_layout_structure(ocr_data)
        
        # Create an output file path by replacing input_hdfs_dir with output_hdfs_dir
        relative_path = ocr_file.replace(input_hdfs_dir, "")
        if relative_path.startswith('/'):
            relative_path = relative_path[1:]
        # Replace '_ocr.json' with '_layout.json'
        output_filename = relative_path.rsplit('_ocr.json', 1)[0] + '_layout.json'
        output_file_path = output_hdfs_dir.rstrip('/') + '/' + output_filename
        
        # Ensure the output directory exists in HDFS
        output_dir = '/'.join(output_file_path.split('/')[:-1])
        try:
            hdfs_client.makedirs(output_dir)
        except Exception as e:
            print(f"Error creating output directory {output_dir}: {e}")
        
        # Write the structured layout JSON to HDFS
        try:
            with hdfs_client.write(output_file_path, encoding='utf-8', overwrite=True) as writer:
                json.dump(layout, writer, indent=4)
            print(f"Structured layout saved to {output_file_path}")
        except Exception as e:
            print(f"Error writing structured layout for {ocr_file}: {e}")

if __name__ == "__main__":
    main()
