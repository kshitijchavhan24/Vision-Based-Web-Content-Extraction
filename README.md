# Vision-Based Web Content Extraction using Transformer Models

An advanced solution that automates full-page screenshot capture and text extraction from web pages, converting unstructured web content into structured data using state-of-the-art transformer models.

## Overview

This project leverages a combination of computer vision and natural language processing to extract and analyze web content. Key functionalities include:
- **Automated Screenshot Capture:** Uses Selenium for full-page screenshot capture.
- **Text Extraction:** Utilizes Tesseract OCR to extract text from images.
- **Data Structuring:** Converts unstructured web content into structured formats for analysis.
- **Embedding Generation:** Transforms extracted text into vector embeddings with SentenceTransformer.
- **Document Indexing:** Indexes 50K+ documents using FAISS for fast similarity searches.
- **RESTful API & Front-End:** Provides scalable RESTful endpoints using FastAPI and an interactive front-end built with Streamlit, integrated with HDFS for storage.

## Features

- **Automated Data Capture:**
  - Full-page screenshot automation using Selenium.
  - Text extraction from images with Tesseract OCR.
- **Content Processing:**
  - Conversion of unstructured data into structured formats.
  - Generation of text embeddings using SentenceTransformer.
  - Efficient indexing and retrieval with FAISS.
- **API and UI Integration:**
  - Scalable RESTful endpoints developed with FastAPI.
  - Interactive front-end built with Streamlit.
  - Integration with HDFS for robust data storage.

## Built With

- **Selenium:** For automated web interaction and screenshot capture.
- **Tesseract OCR:** For optical character recognition and text extraction.
- **SentenceTransformer:** To generate vector embeddings from text.
- **FAISS:** For indexing and similarity search of high-dimensional data.
- **FastAPI:** For developing scalable RESTful APIs.
- **Streamlit:** For creating an interactive front-end.
- **HDFS:** For distributed storage and handling large datasets.
- **Python:** Core language for scripting and pipeline development.

## Getting Started

### Prerequisites

- **Python 3.x:** Ensure Python is installed.
- **Selenium & WebDriver:** For automating browser tasks.
- **Tesseract OCR:** Install Tesseract and configure its path.
- **HDFS:** Setup for distributed file storage (if not using a local alternative).
- Other Python dependencies can be installed using pip (see Installation section).

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/kshitijchavhan24/Vision-Based-Web-Content-Extraction.git
