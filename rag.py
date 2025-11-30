import os
import io
from PIL import Image
from paddleocr import PPStructureV3
from fastapi import FastAPI, File, UploadFile
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


load_dotenv()




app = FastAPI()


# LLM and Embeddings

llm = ChatOpenAI(temperature=0.3, model_name="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small",dimensions=1024)

# Pinecone Vector Store

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "ragtest"
index = pc.Index(index_name)
vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)




paddle_ocr = PPStructureV3(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False
)

splitter = RecursiveCharacterTextSplitter(
                separators=["###", "\n\n", " "],
                chunk_size=500,
                chunk_overlap=100
            )


def extract_with_paddleocr(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    temp_path = "temp_image.png"
    image.save(temp_path)
    output = paddle_ocr.predict(input=temp_path)
    extracted = {
        "text": [],
        "tables": [],
        "markdown": ""
    }
    for res in output:
        layout = res.get("layout_parsing_result", {})

        if "text" in layout:
            extracted["text"].append(layout["text"])
        if "tables" in layout:
            extracted["tables"].extend(layout["tables"])

        res.save_to_markdown(save_path="temp_output")
        for file in os.listdir("temp_output"):
            if file.endswith(".md"):
                file_path = os.path.join("temp_output", file)
                with open(file_path, "r", encoding="utf-8") as f:
                    extracted["markdown"] = f.read()

    if os.path.exists(temp_path):
        os.remove(temp_path)

    return extracted



@app.post("/upload-image/")
async def upload_image_only(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()

        extracted = extract_with_paddleocr(file_bytes)

        with open("summary.md", "w", encoding="utf-8") as f:
            f.write(extracted["markdown"])

        documents = []

        # if the extracted text is text. then we directly make it a doc and add it to tje list
        for t in extracted["text"]:
            if t.strip():
                documents.append(Document(
                    page_content=t,
                    metadata={"type": "text", "source": file.filename}
                ))

        # if the extracted text is markdown. then we split it into chunks and add it to the list
        if extracted["markdown"]:
            
            md_docs = splitter.create_documents([extracted["markdown"]])

            for d in md_docs:
                d.metadata = {"type": "markdown", "source": file.filename}

            documents.extend(md_docs)

        if documents:
            vector_store.add_documents(documents)

        return {
            "message": "File processed successfully",
            "extracted_text_chunks": len(extracted["text"]),
            "tables": len(extracted["tables"]),
            "markdown_chars": len(extracted["markdown"]),
            "documents_stored": len(documents),
        }

    except Exception as e:
        return {"error": str(e)}
    


@app.get("/get-answer/")
async def get_answer(query: str):
    try:
        similar_docs = vector_store.similarity_search(query, k=3)

        prompt = f"""You are an AI assistant that helps users by providing information based on the following context:\n\n{similar_docs}\n\nAnswer the following question:\n{query}"""

        answer = llm.invoke(prompt).content

        return {
            "query": query,
            "answer": answer,
        }

    except Exception as e:
        return {"error": str(e)}