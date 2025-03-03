# Furniture Sales Agent

A RAG-based chatbot for the Mobica Factories catalog. This project leverages advanced Retrieval-Augmented Generation (RAG) techniques using LangChain, FAISS, and Ollama's "aya-expanse:8b" model to deliver detailed and multilingual (English and Arabic) product information from the Mobica catalog.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Future Enhancements](#future-enhancements)
- [License](#license)

---

## Overview

This project implements a sales assistant chatbot that extracts relevant product information from the Mobica Factories PDF catalog. It uses a combination of:

- **LangChain & FAISS:** For efficient vector-based document retrieval.
- **Ollama Embeddings & LLM ("aya-expanse:8b"):** To generate high-quality responses in both Arabic and English.
- **Semantic Chunking:** To split the catalog into contextually meaningful segments.
- **Streamlit:** For an interactive web-based chat interface.

By processing the catalog and creating a vector store, the chatbot can quickly retrieve context-rich segments and generate accurate, context-based answers to user queries.

---

## Features

- **Efficient Retrieval:** Utilizes FAISS and semantic chunking to quickly find the most relevant parts of the catalog.
- **Multilingual Support:** Automatically detects the query language (English or Arabic) and generates responses accordingly.
- **User-Friendly Interface:** Built with Streamlit to provide a seamless interactive chat experience.
- **Robust Conversational Chain:** Custom prompt templates ensure that responses are strictly based on the retrieved catalog data.
- **Performance Optimization:** Caching mechanisms improve system performance by storing processed documents and vector stores.

---

## Installation

### Prerequisites

- Python 3.8 or later
- Git

### Setup Instructions

1. **Clone the repository:**

   git clone https://github.com/Shady-Abdelaziz/Sales-Agent.git  
   cd Sales-Agent

2. **Create and activate a virtual environment (optional but recommended):**

   python -m venv venv  
   (On Windows use: venv\Scripts\activate)

3. **Install the required dependencies:**

   pip install streamlit langchain langchain_community langchain_experimental faiss-cpu pdfplumber langdetect

4. **Configure the Project:**

   - Update the PDF file path in `app.py` (variable `PDF_PATH`) to point to your Mobica catalog PDF file.
   - Ensure that you have proper access to the Ollama API or service if required.

---

## Usage

To run the chatbot, execute the following command in your project directory:

streamlit run app.py

This will launch the Streamlit application in your default web browser. You can then interact with the chatbot by asking questions about the Mobica catalog products. The chatbot will automatically detect your language (English or Arabic) and provide detailed product information accordingly.

---

## Project Structure

Sales-Agent/ │ ├── app.py # Main application file with the chatbot implementation ├── 1561986011.General Catalogue.pdf # PDF catalog (ensure the path in app.py is updated accordingly) ├── requirements.txt # List of required Python packages (if available) └── README.md # This file



---

## Future Enhancements

- **Real-Time Data Integration:** Incorporate real-time inventory and pricing data.
- **Expanded Product Lines:** Extend support to additional catalogs or product types.
- **Enhanced UI/UX:** Further improve the Streamlit interface for a more engaging user experience.
- **Additional Languages:** Expand multilingual support beyond English and Arabic.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

Feel free to explore the code and contribute to further improvements. If you have any questions or feedback, please don't hesitate to reach out!

Happy coding!
