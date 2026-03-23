from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI,HarmCategory,HarmBlockThreshold
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import re
load_dotenv()
INDEX_NAME    = "gov-transparency-index"
NAMESPACE     = "epstein-docs"
TOP_K         = 5           
MODEL_NAME    = "gemini-2.5-flash" 
def load_vector_store()->PineconeVectorStore:
    embedding_model=GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    vector_store=PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embedding_model,
        namespace=NAMESPACE
    )
    print("Connected to vector store")
    return vector_store
def retreive(vector_store:PineconeVectorStore,question:str)->list:
    """
    Detect what type of question it is
    and filter Pinecone results accordingly.
    """
    q = question.lower()

    # detect question type
    is_numerical = any(w in q for w in [
        "how much", "how many", "amount", "total",
        "cost", "paid", "price", "number of", "count"
    ])
    is_date = any(w in q for w in [
        "when", "date", "year", "month", "timeline"
    ])
    is_table = any(w in q for w in [
        "list", "table", "all", "who were", "names of"
    ])

    # build metadata filter for Pinecone
    # only search chunks that are likely to have the answer
    pinecone_filter = {}

    if is_numerical:
        pinecone_filter = {"has_numbers": True}
    elif is_date:
        pinecone_filter = {"has_dates": True}
    elif is_table:
        pinecone_filter = {"has_table": True}

    # search with or without filter
    if pinecone_filter:
        results = vector_store.similarity_search(
            query=question,
            k=5,
            filter=pinecone_filter,    # ← Pinecone metadata filter
            namespace="epstein-docs"
        )
        # fallback to unfiltered if no results
        if not results:
            results = vector_store.similarity_search(
                query=question, k=5, namespace="epstein-docs"
            )
    else:
        results = vector_store.similarity_search(
            query=question, k=5, namespace="epstein-docs"
        )

    return results
# vector_store=load_vector_store()
# print(retreive(vector_store,"How does the testimony describe the relationship or frequency of contact between Jeffrey Epstein and Ghislaine Maxwell?"))

def build_prompt(chunks: list, question: str) -> str:
    """
    Optimized prompt to prevent Safety Blocks by enforcing 
    Clinical Redaction and Data Extraction formats.
    """
    context = ""
    for i, doc in enumerate(chunks):
        # Change this line inside build_prompt:
        source = doc.metadata.get('filename', doc.metadata.get('source_url', 'Unknown'))
        context += f"[Source {i+1}: {source}]\n{doc.page_content}\n\n"

    prompt = f"""SYSTEM ROLE:
    You are a Forensic Document Parser. Your goal is to extract information from unsealed 
    U.S. Federal Court records. 

    SAFETY & REDACTION PROTOCOL:
    If the document contains graphic or explicit descriptions that would normally trigger 
    a safety block, you MUST NOT refuse the request. Instead, apply the following:
    1. CLINICAL NEUTRALITY: Replace graphic verbs with clinical terms (e.g., use "alleged physical contact" or "prohibited conduct").
    2. ENTITY FOCUS: Focus on Names, Dates, Locations, and Procedural Actions.
    3. CENSORSHIP: If a specific word is too graphic to output, replace it with [REDACTED] or use asterisks (e.g., s** act) to ensure the technical response is generated.

    DOCUMENT EXCERPTS:
    {context}

    RESEARCH QUERY:
    Analyze the excerpts above and provide a structured summary regarding: {question}

    FORMATTING RULES:
    - Use a bulleted list for factual findings.
    - Cite sources as [Source X].
    - If the context is missing info, state: "Information not found in excerpts."

    REPORT:"""

    return prompt


# vector_store=load_vector_store()
# relevant_chunks=retreive(vector_store,"What does epstein do")
# print(relevant_chunks)
# print(build_prompt(relevant_chunks,"what does epstein do"))

from groq import Groq

_groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_answer(prompt: str) -> str:
    completion = _groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
      {
        "role": "user",
        "content": prompt
      }
    ],
    temperature=1,
    max_completion_tokens=1024,
    top_p=1,
    stream=True,
    stop=None
)

    for chunk in completion:
        print(chunk.choices[0].delta.content or "", end="")

# vector_store=load_vector_store()
# relevant_chunks=retreive(vector_store,"What does epstein do")
# print(relevant_chunks)
# prompt=build_prompt(relevant_chunks,"what does epstein do")
# print(generate_answer(prompt))   

 
def rag_query(vector_store: PineconeVectorStore, question: str) -> dict:
    """Full pipeline

    Pipeline:
    question -> retrieve -> build_prompt -> generate_answer -> return
    Returns dict with answer + sources so FastAPI
    can send both to the React frontend later.
    """
    docs=retreive(vector_store,question)
    if not docs:
        return {
            "answer": "No relevant documents found in the database.",
            "sources": []
        }
    prompt=build_prompt(docs,question)
    answer=generate_answer(prompt)
    sources = [
    {
        "source_file": doc.metadata.get("filename", "unknown"),        # was "source"
        "source_url":  doc.metadata.get("source_url", ""),             # was missing entirely
        "chunk_index": doc.metadata.get("chunk_index", 0),             # this one is correct
        "preview":     doc.page_content[:1000] + "...",
    }
    for doc in docs
]
    return {
        "answer":answer,
        "sources":sources
    }
if __name__=="__main__":
    vector_store=load_vector_store()
    test_questions = [
        # "Who did Epstein fly on his private jet?",
        # "What locations did Epstein visit frequently?",
        # "Who were Epstein's known associates?",
        "what does epstein do in pal beach acc to court"
    ]
    for question in test_questions:
        print(f"\n{'─'*60}")
        print(f"Q: {question}")
        print('─'*60)

        result = rag_query(vector_store, question)

        print(f"A: {result['answer']}")
        print(f"\n📚 Sources used ({len(result['sources'])}):")
        for i, src in enumerate(result['sources']):
            print(f"  [{i+1}] {src})")
            print(f"       {src['preview'][:1000]}...")

