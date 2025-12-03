import numpy as np
from src.embedding import EmbeddingPipeline
from src.vectorstore import CHROMAVectorStore
from typing import List, Dict, Any, Tuple
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import os

class RAGRetriever:
    def __init__(self, vector_store : CHROMAVectorStore,llm_repo:str='deepseek-ai/DeepSeek-R1'):
        self.embdding_manager=EmbeddingPipeline()
        self.vector_store=vector_store
        load_dotenv()
        token = os.getenv('HF_API_TOKEN')
        endpoint = HuggingFaceEndpoint(repo_id=llm_repo, huggingfacehub_api_token=token)
        self.llm = ChatHuggingFace(llm=endpoint)

    def search(self, query:str , top_k:int = 5):
        results=self.retrieve(query,top_k)
        context = "\n\n".join([doc['content'] for doc in results]) if results else ""
        if not context:
            return "No releveant context found for this query"
        prompt=f"""Use the following context to answer the query
        Context:
        {context}
        
        Question:{query}

        Answer:"""
        response = self.llm.invoke(prompt.format(context=context, query=query))
        return response.content
    
    def retrieve(self, query:str , top_k:int = 5) -> list[Dict[str, Any]]:
        print(f"Retrieving documents for query: '{query}'")

        query_embedding =self.embdding_manager.generate_embeddings([query])[0]
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            retrieved_docs=[]
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                ids = results['ids'][0]
                distances = results['distances'][0]

                for i ,(doc, metadata, id, distance) in enumerate(zip(documents, metadatas, ids, distances)):
                    similarity_score = 1 - distance

                    retrieved_docs.append({
                        'id': id,
                        'content': doc,
                        'metadata': metadata,
                        'similarity_score': similarity_score,
                        'distance': distance,
                        'rank': i+1
                    })
                print(f"Retrieved {len(retrieved_docs)} after filtering...")
            else:
                print("No documents found")

            return retrieved_docs
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []