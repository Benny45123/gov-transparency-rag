from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone,ServerlessSpec
from time import sleep
import pymupdf
import pytesseract
from PIL import Image
from tqdm import tqdm
import os
import json
from dotenv import load_dotenv
load_dotenv()
index_name = "gov-transparency-index"
namespace = "epstein-docs"
chunk_size = 1000
chunk_overlap = 100
ocr_treshold = 50
dpi=300

def extract_text_from_pdf(pdf_path: str)->str:
    """
    - Real text pages  → PyMuPDF (fast)
    - Image-only pages → pytesseract OCR (fallback)
    """
    doc=pymupdf.open(pdf_path)
    full_text=""
    for page_num,page in enumerate(tqdm(doc, desc="Extracting text from PDF")):
        page_text=page.get_text().strip()
        if len(page_text)>ocr_treshold:
            full_text+=f"\n[Page {page_num+1}]\n{page_text}\n"
        else:
            try:
                pix=page.get_pixmap(dpi=dpi)
                img=Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text=pytesseract.image_to_string(img,lang="eng",config="--psm 3").strip()
                if ocr_text:
                    full_text += f"\n[Page {page_num + 1} - OCR]\n{ocr_text}"
                    print(f"     ✓ OCR page {page_num + 1}: {len(ocr_text)} chars")
                else:
                    print(f"     ⚠ Page {page_num + 1}: no text found even with OCR")
            except Exception as e:
                print(f"     ⚠ Error processing page {page_num + 1} with OCR: {e}")
    doc.close()
    return full_text
def chunk_text(text: str,pdf_path: str)->list[dict]:
    splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap,separators=["\n\n","\n","."," ",""])
    chunks=splitter.create_documents([text])
    chunks_with_metadata=[]
    for i,chunk in enumerate(chunks):
        chunks_with_metadata.append({
            "text":chunk.page_content,
            "metadata":{
                "source":pdf_path,
                "chunk_index":i,
                "total_chunks":len(chunks),
                "namespace":namespace
            }
        })
    return chunks_with_metadata
def setup_pinecone_index(pc:Pinecone):
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # 2. Use .names() to get the list of strings to compare against
    existing_indexes = pc.list_indexes().names()
    
    if index_name not in existing_indexes:
        print(f"Creating index {index_name}...")
        pc.create_index(
            name=index_name,
            dimension=3072, 
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    else:
        print(f"Index {index_name} already exists.")
def main():
    for key in ["GEMINI_API_KEY","PINECONE_API_KEY"]:
        if not os.getenv(key):
            raise ValueError(f"Missing env var: {key}")
    pc=Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    embedding_model=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001",google_api_key=os.getenv("GEMINI_API_KEY"))
    setup_pinecone_index(index_name)
    #connect to vectordb(pinecone)
    vector_store=PineconeVectorStore(
        index_name=index_name,
        embedding=embedding_model
    )
    pdf_path="./test-pdfs/Final_Epstein_documents.pdf"
    try:
        raw_text = extract_text_from_pdf(pdf_path)

        if not raw_text.strip():
            print(f"  ⚠ No text extracted — skipping")

        print(f"  ✓ Extracted {len(raw_text)} chars")

        chunks_data = chunk_text(raw_text, pdf_path)
        texts     = [c["text"]     for c in chunks_data]
        metadatas = [c["metadata"] for c in chunks_data]
        print(f"  ✓ {len(texts)} chunks created")
        batch_size=50
        pbar = tqdm(range(0, len(texts), batch_size), desc="Uploading to Pinecone", unit="batch")
        for i in pbar:
            try:
                batch_texts=texts[i:i+batch_size]
                batch_metadatas=metadatas[i:i+batch_size]
                vector_store.add_texts(
                    texts=batch_texts,
                    metadatas=batch_metadatas,
                    namespace=namespace
                )
                sleep(15)
                print(f"batch{i} pushed")
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    pbar.set_description("⏳ Quota hit! Cooling down...")
                    sleep(30)

        print(f"  ✓ Pushed to Pinecone")

    except Exception as e:
        print(f"  ✗ Error processing {pdf_path}: {e}")


    print(f"\n✅ Ingestion complete!")


if __name__ == "__main__":
    main()

    


                
