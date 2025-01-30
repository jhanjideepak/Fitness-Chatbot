import streamlit as st
import pandas as pd
from langchain_groq.chat_models import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from dotenv import load_dotenv
import os
import re
from langchain_community.vectorstores import Chroma
import chromadb
import warnings
from prompts import prompt_template_for_question
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from embed_pdf import create_embeddings_for_new_pdfs, is_query_related
from langchain.schema import HumanMessage, AIMessage


load_dotenv()
warnings.filterwarnings("ignore")

# Get key from env
api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model_name="llama-3.3-70b-specdec", api_key=api_key)

df = pd.read_csv('/Users/deepakjhanji/Downloads/kahunas/MultiRAGChatbot/Fitness-Chatbot/Data/csv_data/dailyActivity_merged.csv')

pdf_directory = "Data/pdf_data"
persist_directory = "pdf_vectordb"

# button to update embeddings in Streamlit app. Using streamlit for GUI.
if st.button("Update Embeddings for New PDFs"):
    create_embeddings_for_new_pdfs(pdf_directory, persist_directory)
    st.write("Embeddings updated with new PDFs.")

chroma_client = chromadb.PersistentClient(path="pdf_vectordb")

# Initialize the LLM
# llm = ChatOpenAI(model="gpt-4", temperature=0)
# Adding module for memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="history", return_messages=True)
    st.session_state.conversation = ConversationChain(
        llm=llm,
        memory=st.session_state.memory,
    )
# Initialize memory
# Access memory and conversation from session state
memory = st.session_state.memory
conversation = st.session_state.conversation
print("Initialized Memory:", memory.chat_memory.messages)
print("Memory object ID:", id(memory))


# Creating a conversation chain with memory
conversation = ConversationChain(
    llm=llm,
    memory=memory,
)

def check_if_data_needed(user_input: str) -> str:
    """
         This module checks the intent of user and then decides whether to fetch pdf embeddings or csv analysis.

         """

    prompt = """The user has asked: "{user_input}".
    Determine the appropriate response based on the question.
    - If the question is related to **user activity or analysis of user health data** and the input query should mention **provide analysis**, respond with "userdata".
    - If it is related to **health, meal and fitness generic questions or experiment result**, respond with "pdfdata"
    - else, respond with **pdfdata** 

    Output should be **one word**: "userdata" or "pdfdata" or  "normal".
    """
    template = PromptTemplate(template=prompt, input_variables=["user_input"])
    chain = template | llm
    decision = chain.invoke({"user_input": user_input}).content.strip().lower()
    print("decision --------------------> ", decision)
    return decision

def get_data_from_csv(user_input: str) -> str:
    prompt_template = """
        The following is a dataset in JSON format:
        {data}

        The user has asked the following question about the dataset:
        "{user_input}"

        Please analyze the data and provide the answer to the query.
        """

    # Fill in the prompt with DataFrame content and the user's query
    df_json = df.to_json(orient="records")

    # Cunking the data to accomodate big file
    def chunk_data(data, chunk_size):
        return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    # Chunking the JSON data
    chunk_size = 1000  # Adjust based on token limits
    data_chunks = chunk_data(df_json, chunk_size)

    responses = []
    for chunk in data_chunks:
        prompt = prompt_template.format(data=chunk, user_input=user_input)

    try:
        # response using the LLM
        response = llm.invoke(prompt)
        responses.append(response.content)
    except Exception as e:
        responses.append(f"Error processing chunk: {str(e)}")

        # Combining all responses and return
    return "\n".join(responses)

    # Generate the response using the LLM
    # output = llm.invoke(prompt)

    # Parse and return the LLM's response
    # return output.content


def respond_to_user(user_input: str):
    """
     Calling the funciton to check user intent and then accordingly make final decisions
     1. If the intent is towards analysis then  ---> read csv file and provide basic analysis
     2. If the intent is towards fitness and health related specific tasks ---> use embeddings from pdf
     """

    decision = check_if_data_needed(user_input)

    if decision == "userdata":
        print("Memory before userdata query:", memory.chat_memory.messages)
        # Use LLM to interpret the user's request for analytics and data analysis
        retrieved_data = get_data_from_csv(user_input)
        # Fetching the prompt with user query and retrieved data from another file
        prompt_template = prompt_template_for_question

        prompt = prompt_template.format(user_input=user_input, retrieved_data=retrieved_data)
        response = conversation.run(input=prompt)

        # Pass the prompt directly as a string to the LLM
        # output = llm.invoke(prompt)

        # Display the response
        # response = output.content
        print("Memory after userdata query:", memory.chat_memory.messages)
        st.write(response)


    elif decision == "pdfdata":

        # Create a pdf vector data base
        persist_directory = "pdf_vectordb"

        # Use embeddings model from langchain_community to create embeddings from pdf files
        embeddings = SentenceTransformerEmbeddings(model_name="all-MPNet-base-v2")
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

        # k represents the number of top documents to retrieve from the vector database in response to a query. Here setting it as 10
        vectorstore_retriever = vectordb.as_retriever(
            search_kwargs={
                "k": 10})

        # chain_type - This concatenates all retrieved documents into a single input string
        # and appends it to the user’s query before sending it to the LLM
        # Using stuff as default but can be changed to map_reduce to handle large documents
        qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                               chain_type="stuff",
                                               retriever=vectorstore_retriever,
                                               return_source_documents=True)

        # Handle user query
        if not memory.chat_memory.messages:
            # First query: process without prior memory
            query = f"You are a helpful AI Assistant. You have expertise in health and fitness domain. Your Job is to generate output based on the query. Requirement: {user_input}"
            llm_response = qa_chain(query)
            st.write(llm_response["result"])
            # Store query and response in memory
            conversation.run(input=user_input)
        else:
            # Check if the query is related to the previous context
            previous_context = "\n".join(
                [
                    f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
                    for msg in memory.chat_memory.messages
                ]
            )
            # previous_context = "\n".join([f"{msg.role}: {msg.content}" for msg in memory.chat_memory.messages])
            if is_query_related(previous_context, user_input):
                print(" Related to previous query")
                # Use memory for related queries
                query = f"You are a helpful AI Assistant. You have expertise in health and fitness domain. " \
                        f"You are continuing a conversation. Requirement: {user_input}"
                llm_response = qa_chain(query)
                st.write(llm_response["result"])
                # Update memory with the new query and response
                conversation.run(input=user_input)
            else:
                # Ignore memory for unrelated queries
                query = f"You are a helpful AI Assistant. You have expertise in health and fitness domain. " \
                        f"Your Job is to generate output based on the query. Requirement: {user_input}"
                llm_response = qa_chain(query)
                st.write(llm_response["result"])
                # Reset memory to start a new conversation
                memory.chat_memory.clear()
                conversation.run(input=user_input)

    else:
        print("Going to else for first query")
        # Generic prompt for first interaction
        prompt = f"""
            The user has asked: "{user_input}".

            You are a helpful assistant. You have expertise in health and fitness domain.
             Please provide a relevant and helpful response to the user's query.
            """

        # Run the conversation or invoke the LLM
        try:
            print("Final Prompt Sent to LLM:", prompt)
            response = llm.invoke(prompt)
            print("Memory after general query:", memory.chat_memory.messages)
            print("Memory object ID:", id(memory))

            st.write(response.content)
        except Exception as e:
            st.write(f"Error processing the query: {str(e)}")


user_input = st.text_input("Enter your query:")
if st.button("Submit"):
    if user_input:
        respond_to_user(user_input)
    else:
        st.warning("Please enter a query.")