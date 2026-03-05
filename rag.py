"""
Simple RAG module for IdeaToEpic
Handles document ingestion, chunking, embedding, and retrieval
"""

from typing import List, Dict
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx
import io


# ─────────────────────────────────────────────
# HELPER FUNCTIONS (text extraction)
# ─────────────────────────────────────────────

def extract_text(file_content: bytes, filename: str) -> str:
    """Extract text from PDF, DOCX, TXT, or MD files"""
    ext = filename.lower().split('.')[-1]
    
    if ext == 'pdf':
        pdf_file = io.BytesIO(file_content)
        reader = PyPDF2.PdfReader(pdf_file)
        return "\n".join([page.extract_text() for page in reader.pages])
    
    elif ext in ['docx', 'doc']:
        doc_file = io.BytesIO(file_content)
        doc = docx.Document(doc_file)
        return "\n".join([para.text for para in doc.paragraphs])
    
    elif ext in ['txt', 'md']:
        return file_content.decode('utf-8', errors='ignore')
    
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(text: str, source: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict]:
    """Split text into overlapping chunks"""
    text = " ".join(text.split())  # Clean whitespace
    
    chunks = []
    start = 0
    chunk_id = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        
        # Don't create tiny final chunks
        if len(chunk_text) < 100 and chunks:
            chunks[-1]['text'] += " " + chunk_text
        else:
            chunks.append({
                'text': chunk_text,
                'source': source,
                'chunk_id': chunk_id
            })
            chunk_id += 1
        
        start += chunk_size - overlap
    
    return chunks


# ─────────────────────────────────────────────
# RAG MANAGER (main class)
# ─────────────────────────────────────────────

class RAGManager:
    """Simple RAG with in-memory vector store"""
    
    def __init__(self, model_name: str = "paraphrase-MiniLM-L3-v2"):
        self.model = SentenceTransformer(model_name)
        self.chunks = []
        self.embeddings = None
    
    def ingest_document(self, file_content: bytes, filename: str) -> int:
        """Process and index a document"""
        # Extract and chunk
        text = extract_text(file_content, filename)
        new_chunks = chunk_text(text, filename)
        
        # Embed
        texts = [c['text'] for c in new_chunks]
        new_embeddings = self.model.encode(texts, show_progress_bar=False)
        
        # Store
        self.chunks.extend(new_chunks)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        return len(new_chunks)
    
    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """Retrieve relevant context for a query"""
        if not self.chunks:
            return ""
        
        # Embed query and compute similarity
        query_embedding = self.model.encode([query], show_progress_bar=False)[0]
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top k chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_chunks = [self.chunks[i] for i in top_indices]
        
        # Format
        context_parts = [f"[From {c['source']}]:\n{c['text']}" for c in top_chunks]
        return "\n\n".join(context_parts)
    
    def get_stats(self) -> Dict:
        """Get statistics about indexed documents"""
        return {
            "total_chunks": len(self.chunks),
            "documents": list(set(c['source'] for c in self.chunks))
        }
    
    def clear(self):
        """Clear all indexed documents"""
        self.chunks = []
        self.embeddings = None