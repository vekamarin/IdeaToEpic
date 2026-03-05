# RAG Integration for IdeaToEpic

## Overview

IdeaToEpic now supports **Document-Enhanced Requirements Generation** using Retrieval Augmented Generation (RAG). Upload reference documents to provide context that informs the requirements pipeline.

## What It Does

When you upload documents (PDFs, DOCX, TXT, MD), the system:
1. Extracts and chunks the text
2. Creates embeddings using sentence-transformers
3. Retrieves relevant chunks during requirements generation
4. Injects this context into the Requirement Architect agent

This allows the pipeline to:
- **Align with existing architecture** (upload tech specs)
- **Reference existing requirements** (upload prior PRDs)
- **Consider constraints** (upload compliance docs)
- **Maintain consistency** (upload style guides)

## Architecture

```
Document Upload → Extract Text → Chunk (500 chars, 100 overlap)
                                  ↓
                              Embed (all-MiniLM-L6-v2)
                                  ↓
                              Store (in-memory numpy)
                                  ↓
Pipeline Start → Retrieve (top 5 chunks based on domain+VOC)
                                  ↓
                    Inject into Architect Prompt
```

**Key Design Decisions:**
- **In-memory storage**: Simple, no external DB needed for demo/MVP
- **Lightweight embeddings**: all-MiniLM-L6-v2 (80MB, fast, good quality)
- **Fixed chunking**: 500 characters with 100 char overlap (simple, works well)
- **Simple retrieval**: Cosine similarity, no reranking (keeps it fast)

## API Usage

### 1. Upload Documents

```bash
# Upload a PDF
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@architecture_spec.pdf"

# Upload multiple files
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@existing_requirements.docx"

curl -X POST http://localhost:8000/documents/upload \
  -F "file=@technical_constraints.md"
```

Response:
```json
{
  "status": "success",
  "filename": "architecture_spec.pdf",
  "chunks_added": 12,
  "stats": {
    "total_chunks": 12,
    "documents": ["architecture_spec.pdf"]
  }
}
```

### 2. Check Uploaded Documents

```bash
curl http://localhost:8000/documents/stats
```

Response:
```json
{
  "total_chunks": 35,
  "documents": [
    "architecture_spec.pdf",
    "existing_requirements.docx",
    "technical_constraints.md"
  ]
}
```

### 3. Generate Requirements (with RAG)

Once documents are uploaded, simply run the pipeline normally:

```bash
curl -X POST http://localhost:8000/generate-stream \
  -H "Content-Type: application/json" \
  -d '{
    "product_domain": "hospital patient scheduling system",
    "generate_voc": true
  }'
```

The pipeline will automatically retrieve relevant chunks from uploaded documents and use them when building requirements.

### 4. Clear Documents

```bash
curl -X DELETE http://localhost:8000/documents/clear
```

## Integration Points

### Where RAG Context is Used

**In `requirement_architect_node`:**
- Retrieved chunks are injected as "Reference Documents" section
- LLM is instructed to:
  - Align with existing technical constraints
  - Reference existing features/requirements
  - Ensure consistency with documented standards
  - Add implementation details where relevant

**Retrieval Query:**
- Combines `product_domain + first 200 chars of VOC`
- Fetches top 5 most relevant chunks
- Formatted with source attribution: `[From filename]: chunk text`

## Example: Using Technical Specs

**Scenario:** You're building a new feature for an existing system with strict performance requirements.

**Without RAG:**
```
User Story: As a nurse, I want to see schedule updates 
so that I can plan my shift.

Acceptance Criteria:
- System displays updated schedules
- Updates appear in real-time
```

**With RAG (uploaded: "performance_requirements.pdf"):**
```
User Story: As a nurse, I want to see schedule updates 
so that I can plan my shift.

Acceptance Criteria:
- System displays updated schedules within 2 seconds (per SLA-001)
- Updates propagate via WebSocket connection (architecture constraint)
- Dashboard refresh rate: max 5 seconds under peak load (1000 concurrent users)
```

The RAG context from your performance requirements doc helped generate **specific, measurable** criteria aligned with existing standards.

## Scaling Up (Future)

Current implementation is MVP-focused. For production:
- Replace in-memory numpy with **FAISS** or **Pinecone** for persistence
- Add **hybrid search** (BM25 + vector similarity)
- Implement **reranking** with cross-encoder
- Use **larger embedding models** (e.g., bge-large-en-v1.5)
- Add **metadata filtering** (by document type, date, etc.)

But start simple — this implementation handles most use cases.

## Dependencies

```bash
pip install sentence-transformers numpy PyPDF2 python-docx
```

All included in `requirements.txt`.

## Testing

See `test_rag.py` for examples of:
- Uploading documents
- Running pipeline with RAG context
- Comparing results with/without RAG
