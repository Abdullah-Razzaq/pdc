from typing import List, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from src.data_loader import load_docs
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
import os
from typing import List, Dict, Any, Tuple

class EmbeddingPipeline:
    def __init__(self, model_name:str = 'Qwen/Qwen3-Embedding-8B', chunk_size: int = 1000, chunk_overlap: int = 200):
        self.model_name = model_name
        self.model = None
        self.chunk_size= chunk_size
        self.chunk_overlap = chunk_overlap
        self._load_model()

    def _load_model(self):
        load_dotenv()
        token = os.getenv("HF_API_TOKEN")
        print(f"Loading embedding model: {self.model_name}")

        self.model = HuggingFaceEndpointEmbeddings(
            repo_id=self.model_name,
            huggingfacehub_api_token=token,
        )
        print("Model loaded successfully....")
    
    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size =self.chunk_size,
            chunk_overlap =self.chunk_overlap,
            separators=['\n\n', '\n', ' ', '']
        ) 
        docs = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(docs)} chunks.")
        return docs
    
    def generate_embeddings(self, documents: list[str] ) -> np.ndarray:
        print(f"Generating embeddings for {len(documents)} documents...")
        embeddings = self.model.embed_documents(documents)
        print(f"Generated embeddings with shape: {np.array(embeddings).shape}")
        return embeddings