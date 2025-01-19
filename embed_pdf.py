from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from pathlib import Path
from langchain.schema import Document
import PyPDF2
from langchain_groq.chat_models import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()
# directory = "Data/pdf_data"
api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model_name="llama-3.1-70b-versatile", api_key=api_key)

def load_pdf_text_pypdf2(file_path):
    """
    Loads PDF file
    """
    text = ""
    with open(file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text


def create_embeddings_for_new_pdfs(directory, persist_directory):
    """
    It store embeddings for pdf files
    """
    documents = []
    for pdf_file in Path(directory).glob("*.pdf"):
        text = load_pdf_text_pypdf2(pdf_file)
        documents.append(Document(page_content=text, metadata={"source": str(pdf_file)}))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)

    embeddings = SentenceTransformerEmbeddings(model_name="all-MPNet-base-v2")
    # persist_directory = "pdf_vectordb"
    vectordb = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)
    vectordb.persist()
    print("Number of stored vectors:", vectordb._collection.count())

    print("PDF documents embedded and saved.")

def is_query_related(previous_context, current_query):
    """
    This checks if its a new query or related to previous context
     :return: Yes or no
     """

    if not previous_context:
        return False  # No previous context, so it's a first query
    prompt = f"""
    Previous query and response:
    {previous_context}

    New user query:
    {current_query}
    
    Rules:
    - If the new query depends on or references the previous conversation, respond "yes".
    - If the new query introduces a completely new topic or does not depend on the previous conversation, respond "no".
    
    Example Scenarios:
    1. Previous Query: "How many students were part of experiment group". New Query: "and were these students chosen at random" --> Respond "Yes".
    2  Previous: "What was the age group of students in experiment" New: "Can you summarize this document" --> Respond "Yes".
    3. Previous: "Can you summarize this document?" New: "What about solar energy?" --> Respond "no".

    Is the new query related to the previous conversation? 
    Respond with "yes" or "no".
    """
    decision = llm.invoke(prompt).content.strip().lower()
    return decision == "yes"

