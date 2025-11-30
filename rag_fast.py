import os
import io
import multiprocessing
from PIL import Image
from paddleocr import PaddleOCR
from fastapi import FastAPI, File, UploadFile
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


load_dotenv()
app = FastAPI()


# LLM + Embeddings
llm = ChatOpenAI(temperature=0.3, model_name="gpt-4o-mini")
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1024
)

# Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("ragtest")
vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

# FAST OCR (not PPStructureV3)
ocr_engine = PaddleOCR(
    use_angle_cls=True,
    lang="en"
)


# ---------------------------------------------------------
#  SMALL HELPER: Run OCR with timeout using subprocess
# ---------------------------------------------------------

def _ocr_worker(image_path, return_dict):
    """Executed in a subprocess — avoids hangs."""
    result = ocr_engine.ocr(image_path)
    return_dict["result"] = result


def run_ocr_with_timeout(image_path, timeout=20):
    """Run PaddleOCR in a safe subprocess with timeout."""
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    p = multiprocessing.Process(
        target=_ocr_worker,
        args=(image_path, return_dict)
    )
    p.start()
    p.join(timeout=timeout)

    if p.is_alive():
        p.terminate()
        return None   # timed out

    return return_dict.get("result", None)


# ---------------------------------------------------------
# Extractor (fast + safe)
# ---------------------------------------------------------

def extract_text_from_image(image_bytes):

    image = Image.open(io.BytesIO(image_bytes))

    temp_path = "temp_input.png"
    image.save(temp_path)

    print("OCR START")
    result = run_ocr_with_timeout(temp_path, timeout=20)
    print("OCR DONE")

    if os.path.exists(temp_path):
        os.remove(temp_path)

    # Timeout or failure
    if result is None:
        return {"text": [], "markdown": "", "tables": []}

    extracted_texts = []
    for line in result:
        for box in line:
            extracted_texts.append(box[1][0])   # box[1] = (text, confidence)

    full_text = "\n".join(extracted_texts)
    print(extracted_texts)

    return {
        "text": extracted_texts,
        "markdown": full_text,
        "tables": []  # plain OCR cannot detect tables, but it's fast
    }


# ---------------------------------------------------------
# FastAPI ROUTES
# ---------------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " "],
    chunk_size=500,
    chunk_overlap=100
)

@app.post("/upload-image/")
async def upload_image_only(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()

        extracted = extract_text_from_image(file_bytes)
        print("EXTRACTED:", extracted)

        documents = []

        # store raw text snippets
        for t in extracted["text"]:
            if t.strip():
                documents.append(
                    Document(
                        page_content=t,
                        metadata={"source": file.filename, "type": "text"}
                    )
                )

        # store markdown chunks
        if extracted["markdown"]:
            md_docs = splitter.create_documents([extracted["markdown"]])
            for d in md_docs:
                d.metadata = {"source": file.filename, "type": "markdown"}
            documents.extend(md_docs)

        # push to Pinecone
        if documents:
            vector_store.add_documents(documents)

        return {
            "message": "File processed",
            "text_chunks": len(extracted["text"]),
            "markdown_chars": len(extracted["markdown"]),
            "documents_stored": len(documents),
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/get-answer/")
async def get_answer(query: str):
    similar_docs = vector_store.similarity_search(query, k=3)
    ctx = "\n\n".join([d.page_content for d in similar_docs])

    prompt = f"""Answer using this context only:

{ctx}

Question: {query}
"""

    answer = llm.invoke(prompt).content
    return {"query": query, "answer": answer}
