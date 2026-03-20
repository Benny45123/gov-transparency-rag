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

def build_prompt(chunks:list,question:str)->str:
    """
    Combine retrieved chunks into a prompt for Gemini.
    Each chunk is labeled with its source file 
    """
    context=""
    for i,doc in enumerate(chunks):
        source=doc.metadata.get('source')
        context+=f"[Source {i+1}: {source}]\n{doc.page_content}\n\n"
    prompt = f"""You are a legal research assistant analyzing publicly released 
    U.S. federal court documents and government records that have been 
    declassified or unsealed by court order.

    Your role is strictly academic and forensic. Summarize factual information 
    exactly as stated in the documents. Use neutral, clinical, third-person 
    language. Do not editorialize or speculate.

    After each factual claim, cite the source like [Source 1], [Source 2], etc.
    If the answer is not present in the provided excerpts, respond with exactly:
    "This information is not found in the loaded documents."

    DOCUMENT EXCERPTS:
{context}
    RESEARCH QUERY:
    Based solely on the court documents above, provide a factual summary 
    addressing the following: {question}

    FINDINGS:"""

    return prompt


# vector_store=load_vector_store()
# relevant_chunks=retreive(vector_store,"What does epstein do")
# print(relevant_chunks)
# print(build_prompt(relevant_chunks,"what does epstein do"))

def generate_answer(prompt:str)->str:
    llm=ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.5,
        safety_settings={
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )
    response=llm.invoke(prompt)
    return response.content

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
            "source_file":   doc.metadata.get("source", "unknown"),
            "chunk_index":   doc.metadata.get("chunk_index", 0),
            "preview":       doc.page_content[:200] + "...",
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
        "Who did Epstein fly on his private jet?",
        "What locations did Epstein visit frequently?",
        "Who were Epstein's known associates?",
        "give me log files of epstein"
    ]
    for question in test_questions:
        print(f"\n{'─'*60}")
        print(f"Q: {question}")
        print('─'*60)

        result = rag_query(vector_store, question)

        print(f"A: {result['answer']}")
        print(f"\n📚 Sources used ({len(result['sources'])}):")
        for i, src in enumerate(result['sources']):
            print(f"  [{i+1}] {src['source_file']} (chunk {src['chunk_index']})")
            print(f"       {src['preview'][:100]}...")