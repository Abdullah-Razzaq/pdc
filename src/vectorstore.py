import os
import numpy as np
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Tuple
from src.embedding import EmbeddingPipeline

class CHROMAVectorStore:
    def __init__(self, collection_name:str='legal_texts_pipeline', persistent_directory:str='./src/vectorStore', 
                 embedding_model:str='Qwen/Qwen3-Embedding-8B', chunk_size: int = 800, chunk_overlap: int = 200):
        self.collection_name=collection_name
        self.persistent_directory=persistent_directory
        self.client=None
        self.collection=None
        self.embedding_model= embedding_model
        self.chunk_size=chunk_size
        self.chunk_overlap =chunk_overlap
        self._initialize_store()

    def _initialize_store(self):
        print("Initialize ChromaDB client and collection")

        os.makedirs(name=self.persistent_directory, exist_ok=True)
        ##Make a new client 
        self.client = chromadb.PersistentClient(path=self.persistent_directory)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={'Description': 'Basic embeddings for Legal Docs pipeline'}
        )
        print(f"Vector store initialized. Collection: {self.collection_name}")
        print(f"Existing documents in collection: {self.collection.count()}")
    
    def load(self):
        print(f"Vector store loaded. Collection: {self.collection_name}")
        print(f'Existing documents in collection {self.collection.count()}')

    def load_and_add_docs(self, documents: list[Any]):
        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = emb_pipe.chunk_documents(documents)
        texts = [text.page_content for text in chunks]
        embeddings = emb_pipe.generate_embeddings(texts)
        self.add_documents(documents=chunks, embeddings=embeddings)

    def add_documents(self, documents: list[Any], embeddings: np.ndarray):
        print(f'Adding documents and embeddings in {self.collection_name}')

        if(len(documents) != len(embeddings)):
            raise ValueError("Document must match the size of embeddings")
        
        ids=[]
        metadatas=[]
        document_texts= []
        embedding_list=[]

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generate a unique ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            # Prepare Metadata
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

             # Document content
            document_texts.append(doc.page_content)

            # Embedding
            embedding_list.append(embedding)

        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embedding_list,
                documents=document_texts,
                metadatas=metadatas
            )

            print(f"Successfully added {len(documents)} to vector store")
            print(f'Existing documents in collection {self.collection.count()}')
        except Exception as e:
            print(f'Error adding documents to vector store {e}')
            raise