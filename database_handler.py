"""
VectorDatabaseHandler

This class handles the creation and loading of vector databases from PDF documents using Google's Gemini embeddings and ChromaDB.
It supports creating a new vector store from a list of PDF paths and loading it for retrieval with a customizable search configuration.

Functions:
- create_database(): Generates the vector database if it does not already exist.
- load_database(): Loads the vector database and returns a configured retriever and LLM for RAG.

Author: [Your Name]
"""

import os
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain import hub

class VectorDatabaseHandler:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    def create_database(self, database_path: str, pdf_paths: List[str]) -> None:
        """
        Creates a vector database from PDF documents if it does not already exist.

        Args:
            database_path (str): Path where the vector database will be saved.
            pdf_paths (List[str]): List of paths to the PDF documents.
        """
        persist_directory = Path(database_path)
        if persist_directory.exists() and any(persist_directory.iterdir()):
            print(f"✅ Vector database found at '{persist_directory}', using existing data.")
            return

        print(f"🟡 Vector database not found at '{persist_directory}'. Creating a new one...")

        # Load and split documents
        documents = []
        for path in pdf_paths:
            loader = PyPDFLoader(path)
            documents.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(documents)

        # Create vector store and persist
        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            persist_directory=str(persist_directory)
        )
        print(f"🟢 Vector database successfully created and saved at '{persist_directory}'.")

    def load_database(self,
                      database_path: str,
                      gemini_model: str = "gemini-2.0-flash",
                      search_type: str = "similarity",
                      search_kwargs: Optional[dict] = None,
                      temperature: float = 0):
        """
        Loads an existing vector database and sets up the retriever and LLM for RAG.

        Args:
            database_path (str): Path to the persisted vector database.
            gemini_model (str): Gemini model to use (default: "gemini-2.0-flash").
            search_type (str): Type of search used by the retriever (default: "similarity").
            search_kwargs (dict): Parameters for the retriever (default: {"k": 10}).
            temperature (float): Temperature for the language model (default: 0).

        Returns:
            retriever, llm, prompt: Configured retriever, LLM, and RAG prompt.
        """
        if search_kwargs is None:
            search_kwargs = {"k": 10}

        print(f"🟡 Loading vector database from '{database_path}'...")
        retriever = Chroma(
            persist_directory=database_path,
            embedding_function=self.embeddings
        ).as_retriever(search_type=search_type, search_kwargs=search_kwargs)

        llm = ChatGoogleGenerativeAI(model=gemini_model, temperature=temperature)
        prompt = hub.pull("rlm/rag-prompt")

        print("🟢 Vector database loaded successfully.")
        return retriever, llm, prompt

############ EXAMPLE USAGE ############
# Instantiate the class
# vdb = VectorDatabaseHandler()

# # Create vector DB
# vdb.create_database(
#     database_path="Databases/RehabilitationDoctorDocs",
#     pdf_paths=[
#         "BibliografiaRAG/Cable_driven_exoskeleton_for_ankle_rehabilitation_in_children_with_cerebral_palsy.pdf",
#         "BibliografiaRAG/Discover2Walk_A_cable-driven_robotic_platform_to_promote_gait_in_pediatric_population.pdf",
#     ]
# )

# # Load vector DB
# retriever, llm, prompt = vdb.load_database(
#     database_path="Databases/RehabilitationDoctorDocs",
#     gemini_model="gemini-2.0-flash",
#     search_kwargs={"k": 5},
#     temperature=0
# )
