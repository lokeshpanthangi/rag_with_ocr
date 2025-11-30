# RAG with Image OCR

This project demonstrates Retrieval-Augmented Generation (RAG) using OCR (Optical Character Recognition) on images, powered by FastAPI, PaddleOCR, LangChain, and Pinecone. It supports extracting text and tables from images, storing them as vector embeddings, and answering queries using LLMs.

## Features
- Extract text and tables from images using PaddleOCR
- Store extracted content in Pinecone vector database
- Query stored content using OpenAI LLMs via LangChain
- FastAPI endpoints for image upload and Q&A

---

## Step-by-Step Setup & Usage

### 1. Clone the Repository
```powershell
git clone https://github.com/lokeshpanthangi/rag_with_ocr.git
cd rag_with_ocr/rag_image_ocr
```

### 2. Create & Activate Python Virtual Environment
```powershell
python -m venv venv
& ./venv/Scripts/Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root with the following keys:
```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

### 5. Start Pinecone Index
- Create a Pinecone index named `ragtest` (dimension: 1024, metric: cosine, pod_type: p1).
- You can do this via the Pinecone dashboard or API.

### 6. Run the FastAPI Server
Choose which implementation to run:
- **Fast OCR (rag_fast.py):**
  ```powershell
  uvicorn rag_fast:app --reload
  ```
- **Document Structure OCR (rag.py):**
  ```powershell
  uvicorn rag:app --reload
  ```

### 7. Use the API
#### Upload an Image
Send a POST request to `/upload-image/` with an image file:
```bash
curl -X POST "http://127.0.0.1:8000/upload-image/" -F "file=@your_image.png"
```
#### Query the Extracted Content
Send a GET request to `/get-answer/?query=your_question`:
```bash
curl "http://127.0.0.1:8000/get-answer/?query=What is in the image?"
```

---

## File Overview
- `rag_fast.py`: Fast OCR pipeline (plain text extraction)
- `rag.py`: Advanced OCR pipeline (text, tables, markdown)
- `requirements.txt`: Python dependencies
- `README.md`: This guide
- `temp_output/`: Stores temporary markdown and images

---

## Troubleshooting
- Ensure your API keys are correct and active.
- Pinecone index must be created before running the server.
- For GPU acceleration, install PaddlePaddle with CUDA support (see [PaddlePaddle docs](https://www.paddlepaddle.org.cn/install/quick)).
- If you encounter import errors, check your Python version (recommended: 3.10+).

---

## References
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [LangChain](https://github.com/langchain-ai/langchain)
- [Pinecone](https://www.pinecone.io/)
- [FastAPI](https://fastapi.tiangolo.com/)

---
