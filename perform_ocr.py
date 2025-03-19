# perform_ocr.py

import json
from io import BytesIO
import pytesseract
from pytesseract import Output
from PIL import Image
from hdfs import InsecureClient

def perform_ocr_on_image(image_bytes):
    """
    Perform OCR on the given image bytes and return a list of detected words along with their bounding boxes.
    """
    image = Image.open(BytesIO(image_bytes))
    data = pytesseract.image_to_data(image, output_type=Output.DICT)
    
    words = []
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        text = data['text'][i].strip()
        if text:  # Only consider non-empty text
            word_info = {
                'text': text,
                'left': data['left'][i],
                'top': data['top'][i],
                'width': data['width'][i],
                'height': data['height'][i],
                'confidence': data['conf'][i]
            }
            words.append(word_info)
    return words

def list_hdfs_files(client, hdfs_path):
    """
    Recursively lists all files (with their full HDFS paths) within a given HDFS directory.
    """
    file_paths = []
    try:
        # List items in the directory with their status information.
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
    # Initialize the HDFS client (adjust the URL, user, and timeout as needed)
    hdfs_client = InsecureClient('http://localhost:9870', user='kc', timeout=300)
    
    # Define the HDFS directories for input screenshots and OCR output
    input_hdfs_dir = '/scraped_images'
    output_hdfs_dir = '/ocr_output'
    
    # Ensure the output directory exists on HDFS
    try:
        hdfs_client.makedirs(output_hdfs_dir)
    except Exception as e:
        print(f"Output directory creation note: {e}")
    
    # Recursively list all PNG files in the input directory
    all_files = list_hdfs_files(hdfs_client, input_hdfs_dir)
    png_files = [f for f in all_files if f.lower().endswith('.png')]
    
    if not png_files:
        print("No PNG files found in HDFS directory:", input_hdfs_dir)
        return
    
    for file_path in png_files:
        print(f"Processing {file_path}...")
        try:
            with hdfs_client.read(file_path) as reader:
                image_bytes = reader.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue
        
        # Perform OCR on the image bytes
        words = perform_ocr_on_image(image_bytes)
        
        # Construct a relative path for output based on the input path.
        # For example, if file_path is:
        #   /scraped_images/url_folder/page_1/screenshot_1.png
        # then the OCR output will be saved as:
        #   /ocr_output/url_folder/page_1/screenshot_1_ocr.json
        relative_path = file_path.replace(input_hdfs_dir, "")
        if relative_path.startswith('/'):
            relative_path = relative_path[1:]
        json_filename = relative_path.rsplit('.', 1)[0] + '_ocr.json'
        output_file_path = output_hdfs_dir.rstrip('/') + '/' + json_filename
        
        # Create any necessary directories in the output HDFS path
        output_dir = '/'.join(output_file_path.split('/')[:-1])
        try:
            hdfs_client.makedirs(output_dir)
        except Exception as e:
            print(f"Error creating output directory {output_dir}: {e}")
        
        # Write the OCR result as a JSON file to HDFS
        try:
            with hdfs_client.write(output_file_path, overwrite=True, encoding='utf-8') as writer:
                json.dump(words, writer, indent=4)
            print(f"OCR data saved to {output_file_path}")
        except Exception as e:
            print(f"Error writing OCR data for {file_path}: {e}")

if __name__ == "__main__":
    main()
