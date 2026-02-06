import os
import requests
import pdfplumber
import numpy as np
import faiss
from openai import OpenAI
import pandas as pd

class RAGPipeline:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
        self.index = None
        self.chunks = []
        
        # Model IDs (Switched to Stable Hosted Versions)
        self.EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
        self.RERANK_MODEL = "nvidia/nv-rerankqa-mistral-4b-v3"
        self.CHAT_MODEL = "nvidia/llama-3.3-nemotron-super-49b"

    def extract_text(self, pdf_file):
        """Extracts text and tables from a PDF using pdfplumber."""
        full_text = []
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                # Extract text
                text = page.extract_text()
                if text:
                    full_text.append(text)
                
                # Extract tables (simple version)
                tables = page.extract_tables()
                for table in tables:
                    # Convert list of lists to markdown table string
                    df = pd.DataFrame(table[1:], columns=table[0])
                    markdown_table = df.to_markdown(index=False)
                    full_text.append(f"\n[TABLE FOUND]\n{markdown_table}\n")
        
        # Simple chunking for demo purposes
        joined_text = "\n\n".join(full_text)
        chunk_size = 1000
        chunks = [joined_text[i:i+chunk_size] for i in range(0, len(joined_text), chunk_size)]
        return chunks

    def create_embeddings(self, chunks):
        """Generates embeddings for the provided chunks using NVIDIA API (Direct Request)."""
        if not chunks:
            return None
        
        # Exact Invoke URL for nv-embedqa-e5-v5
        invoke_url = "https://ai.api.nvidia.com/v1/retrieval/nvidia/nv-embedqa-e5-v5"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        payload = {
            "input": chunks,
            "input_type": "passage",
            "model": "passage"
        }
        
        try:
            response = requests.post(invoke_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            body = response.json()
            embeddings = [item['embedding'] for item in body['data']]
            return np.array(embeddings).astype('float32')
            
        except Exception as e:
            print(f"Embedding API failed. URL: {invoke_url}")
            if 'response' in locals():
                 print(f"Response: {response.text}")
            raise e

    def build_index(self, chunks):
        """Builds a FAISS index from text chunks."""
        self.chunks = chunks
        embeddings = self.create_embeddings(chunks)
        
        if embeddings is None:
            return False
            
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        return True

    def retrieve(self, query, k=5):
        """Retrieves top-k relevant chunks."""
        if self.index is None:
            return []
            
        # Need to embed the query too
        invoke_url = "https://ai.api.nvidia.com/v1/retrieval/nvidia/nv-embedqa-e5-v5"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        payload = {
            "input": [query],
            "input_type": "query",
            "model": "query"
        }
        
        response = requests.post(invoke_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        query_embedding = response.json()['data'][0]['embedding']
        
        query_vector = np.array([query_embedding]).astype('float32')
        
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results.append({
                    "content": self.chunks[idx],
                    "raw_score": float(distances[0][i])
                })
        return results

    def rerank(self, query, retrieved_docs):
        """Reranks retrieved documents using NVIDIA Rerank API."""
        if not retrieved_docs:
            return []

        invoke_url = "https://ai.api.nvidia.com/v1/retrieval/nvidia/nv-rerankqa-mistral-4b-v3"
        
        payload = {
            "model": self.RERANK_MODEL,
            "query": {"text": query},
            "documents": [{"text": doc['content']} for doc in retrieved_docs]
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(invoke_url, json=payload, headers=headers)
            response.raise_for_status()
            rankings = response.json().get('rankings', [])
            
            # Reorder based on new rankings
            reranked = []
            for rank in rankings:
                idx = rank['index']
                doc = retrieved_docs[idx]
                doc['score'] = rank['logit']
                reranked.append(doc)
            
            # Sort by score descending
            reranked.sort(key=lambda x: x['score'], reverse=True)
            return reranked
        except Exception as e:
            print(f"Reranking failed: {e}")
            return retrieved_docs # Fallback to original order

    def generate_response(self, query, context_chunks):
        """Generates a response using the LLM with retrieved context."""
        context_text = "\n\n".join([c['content'] for c in context_chunks])
        
        system_prompt = (
            "You are a helpful assistant. Use the following context to answer the user's question. "
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context_text}"
        )
        
        completion = self.client.chat.completions.create(
            model=self.CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024,
            stream=True
        )
        return completion
