# Health and Fitness Chatbot

This application is designed to provide insightful responses to fitness-related queries using language models. The chatbot is built using Streamlit, making it easy to interact with and explore various functionalities.

## Project Overview

The Fitness Chatbot leverages a combination of language models, data processing, and vector embeddings to deliver accurate and context-aware responses. It integrates with various data sources including CSV files and PDF documents.

## Key Features

- **Interactive User Interface**: Built with Streamlit.
- **Data-Driven Insights**: Utilizes Pandas to process and analyze fitness data from CSV files.
- **Advanced Language Models**: Integrates with the `langchain_groq` library to process natural language queries.
- **Memory Management**: Maintains conversation context using a conversation buffer memory.
- **PDF Embeddings**: Supports embedding and retrieval of information from PDF documents using Chroma.

## Installation

To get started with the Fitness Chatbot, follow these steps:

1. **Clone the Repository**: 
   ```bash
   git clone https://github.com/jhanjideepak/Fitness-Chatbot.git
   cd Fitness-Chatbot
   ```

2. **Install Dependencies**: 
   Ensure you have Python >=3.11 installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**: 
   Create a `.env` file in the project root and add your API keys:
   ```
   GROQ_API_KEY=your_api_key_here
   ```

## Running the App

To launch the Streamlit app, execute the following command in your terminal:
```bash
streamlit run app.py
```

## Design Choices

- **Streamlit**: Chosen for its simplicity and ability to quickly deploy interactive web applications.
- **Pandas**: Used for its powerful data manipulation capabilities, essential for processing fitness data.
- **Langchain**: Provides a robust framework for integrating language models, enhancing the chatbot's ability to understand and respond to queries.
- **Chroma**: Utilized for efficient storage and retrieval of vector embeddings, crucial for handling vectors.

