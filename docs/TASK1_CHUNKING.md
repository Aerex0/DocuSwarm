# Task 1: Financial Document Chunking and Storage

This document details the approach for parsing, chunking, and storing financial documents to optimize retrieval for downstream query answering.

---

## Table of Contents

1. [Overview](#overview)
2. [Document Parsing Strategy](#document-parsing-strategy)
3. [Chunking Strategies](#chunking-strategies)
4. [Multimodal Content Handling](#multimodal-content-handling)
5. [Storage & Indexing](#storage--indexing)
6. [Retrieval Optimization](#retrieval-optimization)
7. [Implementation Details](#implementation-details)

---

## Overview

Financial documents (10-K reports, earnings statements) present unique challenges:
- **Length**: 100+ pages per document
- **Structure**: Complex hierarchical sections
- **Multimodal**: Text, tables, figures, charts
- **Context**: Important relationships between sections

**Goal**: Create a chunking and storage system that:
1. Preserves document structure and context
2. Handles multimodal content effectively
3. Enables fast, accurate retrieval for complex queries
4. Supports reasoning across multiple chunks

---

## Document Parsing Strategy

### Two-Stage Parsing Pipeline

```
┌──────────────┐
│   PDF File   │
│  (10-K Report)
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────┐
│  STAGE 1: LlamaParse (Primary)      │
│  ✓ Complex layout analysis          │
│  ✓ Table extraction (high accuracy) │
│  ✓ Figure detection & captioning    │
│  ✓ Preserves structure              │
└──────┬──────────────────────────────┘
       │ success
       ▼
┌─────────────────────────────────────┐
│  Structured Output (Markdown)       │
│  - Sections with headers            │
│  - Tables as JSON                   │
│  - Figure placeholders + captions   │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  STAGE 2: Post-Processing           │
│  - Extract metadata                 │
│  - Validate table structures        │
│  - Link figures to context          │
└─────────────────────────────────────┘

       │ (fallback if LlamaParse fails)
       ▼
┌─────────────────────────────────────┐
│  Fallback: PyMuPDF + Camelot        │
│  - Basic text extraction            │
│  - Simple table extraction          │
│  - Page-level processing            │
└─────────────────────────────────────┘
```

### LlamaParse Configuration

```python
from llama_parse import LlamaParse

parser = LlamaParse(
    api_key=os.getenv("LLAMAPARSE_API_KEY"),
    result_type="markdown",  # Structured output format
    use_vendor_multimodal_model=True,  # Enhanced table/figure extraction
    vendor_multimodal_model_name="anthropic-sonnet-3.5",
    parsing_instruction="""
    This is a financial 10-K report. Please:
    1. Preserve section hierarchy (Item 1, Item 1A, etc.)
    2. Extract all tables with headers and data
    3. Identify charts/figures with captions
    4. Maintain page numbers for reference
    """,
    max_timeout=300
)

# Parse document
documents = parser.load_data("AMAZON_2022_10K.pdf")
```

**Output Structure:**
```markdown
# Item 1. Business
Amazon.com, Inc. provides online retail services...

## Table: Net Sales by Segment (in millions)
| Year | North America | International | AWS | Total |
|------|---------------|---------------|-----|-------|
| 2022 | $315,880      | $118,007      | $80,096 | $513,983 |
| 2021 | $279,833      | $127,787      | $62,202 | $469,822 |

![Figure 1: Revenue Growth Trend]
Caption: Year-over-year revenue growth across segments
```

### PyMuPDF Fallback (Simple Documents)

```python
import fitz  # PyMuPDF

def fallback_parse(pdf_path):
    """Fallback parser for simple documents"""
    doc = fitz.open(pdf_path)
    
    sections = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        
        # Extract tables using Camelot (if detected)
        tables = camelot.read_pdf(
            pdf_path,
            pages=str(page_num),
            flavor='stream'
        )
        
        sections.append({
            "page": page_num,
            "text": text,
            "tables": [t.df.to_dict() for t in tables]
        })
    
    return sections
```

---

## Chunking Strategies

We employ a **three-level chunking hierarchy** to balance context preservation and retrieval precision.

### 1. Semantic Chunking

**Goal**: Group semantically related content together

**Approach**:
- Embed sentences using Groq's Nomic Embed model
- Compute embedding similarity between adjacent sentences
- Create chunk boundaries where similarity drops below threshold

```python
from sentence_transformers import util

def semantic_chunking(text, max_chunk_size=1024, similarity_threshold=0.7):
    """Chunk text based on semantic similarity"""
    
    sentences = split_into_sentences(text)
    embeddings = embed_sentences(sentences)  # Using Groq
    
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        similarity = util.cos_sim(
            embeddings[i-1], 
            embeddings[i]
        ).item()
        
        # Start new chunk if similarity drops or size exceeded
        if similarity < similarity_threshold or \
           len(' '.join(current_chunk)) > max_chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])
    
    chunks.append(' '.join(current_chunk))
    return chunks
```

**Advantages**:
- Preserves topic coherence
- Natural chunk boundaries
- Better retrieval for conceptual queries

**Disadvantages**:
- May miss structural context
- Computationally expensive

### 2. Structure-Aware Chunking

**Goal**: Preserve document hierarchy and section boundaries

**Approach**:
- Parse markdown headers (# Item 1, ## Risk Factors)
- Create chunks that respect section boundaries
- Maintain parent-child relationships

```python
def structure_aware_chunking(markdown_text, max_chunk_size=1024):
    """Chunk while respecting document structure"""
    
    sections = parse_markdown_sections(markdown_text)
    chunks = []
    
    for section in sections:
        header = section['header']
        content = section['content']
        level = section['level']  # H1, H2, H3, etc.
        
        # Small sections: keep as single chunk
        if len(content) < max_chunk_size:
            chunks.append({
                'text': f"{header}\n\n{content}",
                'header_path': section['path'],  # e.g., "Item 1 > Business > Overview"
                'level': level
            })
        else:
            # Large sections: split but preserve header context
            sub_chunks = split_text(content, max_chunk_size)
            for i, sub_chunk in enumerate(sub_chunks):
                chunks.append({
                    'text': f"{header} (Part {i+1})\n\n{sub_chunk}",
                    'header_path': section['path'],
                    'level': level,
                    'part': i+1
                })
    
    return chunks
```

**Chunk Metadata Example**:
```json
{
  "chunk_id": "uuid-123",
  "text": "Item 1A. Risk Factors...",
  "header_path": "Item 1A > Risk Factors > Operational Risks",
  "level": 2,
  "section": "Risk Factors",
  "page_start": 15,
  "page_end": 17
}
```

**Advantages**:
- Preserves document structure
- Easy to navigate hierarchy
- Natural boundaries for reasoning

### 3. Multimodal Chunking

**Goal**: Integrate text, tables, and figures into coherent chunks

**Approach**:
- Keep tables with surrounding text context
- Link figures to relevant paragraphs
- Create hybrid representations

```python
def multimodal_chunking(parsed_document):
    """Create chunks with text, tables, and figures"""
    
    chunks = []
    
    for section in parsed_document['sections']:
        # Extract components
        text_blocks = section['text_blocks']
        tables = section['tables']
        figures = section['figures']
        
        # Strategy 1: Table-centric chunks
        for table in tables:
            context_before = get_preceding_text(text_blocks, table['position'])
            context_after = get_following_text(text_blocks, table['position'])
            
            chunk = {
                'content_type': 'table',
                'table_data': table['data'],
                'table_markdown': table['markdown'],
                'text_description': generate_table_description(table),
                'context_before': context_before,
                'context_after': context_after,
                'embedding': embed_multimodal(table, context)
            }
            chunks.append(chunk)
        
        # Strategy 2: Text chunks with figure references
        for text_block in text_blocks:
            if has_figure_reference(text_block):
                figure = find_referenced_figure(text_block, figures)
                chunk = {
                    'content_type': 'text_with_figure',
                    'text': text_block,
                    'figure_caption': figure['caption'],
                    'figure_url': figure['url'],
                    'embedding': embed_text(text_block + figure['caption'])
                }
                chunks.append(chunk)
    
    return chunks
```

### Table Representation Strategy

**Hybrid Representation** (both for retrieval and reasoning):

1. **Structured JSON**: Preserve table structure for computation
```json
{
  "headers": ["Year", "Revenue", "Growth"],
  "rows": [
    ["2021", "$469.8B", "21.7%"],
    ["2022", "$514.0B", "9.4%"]
  ]
}
```

2. **Markdown Text**: Enable semantic search
```markdown
Revenue table showing Year-over-Year growth:
- 2021: Revenue $469.8B, Growth 21.7%
- 2022: Revenue $514.0B, Growth 9.4%
```

3. **Natural Language Description**: Improve retrieval
```
This table presents Amazon's annual revenue from 2021 to 2022, 
showing a revenue increase from $469.8 billion to $514.0 billion, 
representing a growth rate decline from 21.7% to 9.4%.
```

**All three representations are embedded and stored** for different query types.

---

## Multimodal Content Handling

### Tables

**Extraction Process**:
1. LlamaParse extracts table with high accuracy
2. Convert to pandas DataFrame for validation
3. Generate three representations (JSON, Markdown, NL)
4. Store with metadata (header row, data types, units)

**Storage Schema**:
```python
{
    "chunk_id": "table_xyz",
    "content_type": "table",
    "table_json": {...},
    "table_markdown": "...",
    "table_description": "...",
    "table_metadata": {
        "title": "Net Sales by Segment",
        "columns": ["Year", "North America", "International", "AWS"],
        "units": "millions USD",
        "years": [2021, 2022],
        "page": 45
    },
    "embedding": [0.1, 0.2, ...],  # Embedded from description
    "context_text": "..."  # Surrounding paragraphs
}
```

### Figures & Charts

**Handling Strategy**:
1. **Caption Extraction**: LlamaParse extracts figure captions
2. **Context Linking**: Associate figures with surrounding text
3. **Visual Description**: (Optional) Use multimodal LLM to describe chart
4. **Reference Tracking**: Track all text references to figure

**Storage Schema**:
```python
{
    "chunk_id": "figure_abc",
    "content_type": "figure",
    "figure_caption": "Figure 1: Revenue Growth Trend",
    "figure_description": "Bar chart showing quarterly revenue growth...",
    "figure_url": "s3://bucket/figures/fig1.png",
    "referenced_by": ["chunk_123", "chunk_456"],
    "context_text": "As shown in Figure 1, revenue growth accelerated...",
    "embedding": [...]
}
```

### Cross-Modal Linking

**Challenge**: Maintain relationships between text, tables, and figures

**Solution**: Create a knowledge graph structure

```python
{
    "chunk_id": "text_789",
    "text": "Our revenue grew 22% in 2021 (see Table 3)...",
    "links": {
        "tables": ["table_xyz"],  # Link to Table 3
        "figures": ["figure_abc"],  # Link to related figure
        "related_chunks": ["text_790", "text_791"]  # Related text
    }
}
```

---

## Storage & Indexing

### ChromaDB Setup

```python
import chromadb
from chromadb.config import Settings

# Initialize ChromaDB client
client = chromadb.PersistentClient(
    path="./data/chromadb",
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=True
    )
)

# Create collection with custom embedding function
from src.utils.groq_client import GroqEmbeddings

embedding_function = GroqEmbeddings(
    model="nomic-embed-text",
    api_key=os.getenv("GROQ_API_KEY")
)

collection = client.create_collection(
    name="financial_documents",
    embedding_function=embedding_function,
    metadata={
        "description": "Amazon 10-K reports 2015-2022",
        "chunking_strategy": "multimodal",
        "embedding_model": "nomic-embed-text"
    }
)
```

### Indexing Process

```python
def index_chunks(chunks, collection):
    """Index chunks in ChromaDB with metadata"""
    
    for chunk in chunks:
        # Prepare document text (different based on content type)
        if chunk['content_type'] == 'text':
            doc_text = chunk['text']
        elif chunk['content_type'] == 'table':
            doc_text = chunk['table_description'] + " " + chunk['table_markdown']
        elif chunk['content_type'] == 'figure':
            doc_text = chunk['figure_caption'] + " " + chunk['context_text']
        
        # Add to ChromaDB
        collection.add(
            ids=[chunk['chunk_id']],
            documents=[doc_text],
            metadatas=[{
                'document_id': chunk['document_id'],
                'content_type': chunk['content_type'],
                'section': chunk['section'],
                'page_start': chunk['page_start'],
                'page_end': chunk['page_end'],
                'year': extract_year(chunk['document_id']),
                'header_path': chunk.get('header_path', ''),
                'word_count': len(doc_text.split())
            }]
        )
```

### Metadata Filtering

**Supported Filters**:
```python
# Filter by document year
results = collection.query(
    query_texts=["revenue growth 2022"],
    where={"year": 2022},
    n_results=5
)

# Filter by content type (only tables)
results = collection.query(
    query_texts=["financial metrics"],
    where={"content_type": "table"},
    n_results=10
)

# Filter by section
results = collection.query(
    query_texts=["risk factors"],
    where={"section": "Risk Factors"},
    n_results=5
)

# Complex filtering (year range + section)
results = collection.query(
    query_texts=["revenue trends"],
    where={
        "$and": [
            {"year": {"$gte": 2020}},
            {"section": "MD&A"}
        ]
    },
    n_results=10
)
```

---

## Retrieval Optimization

### Hybrid Retrieval Strategy

**Combining semantic search (vector) + keyword search (BM25)**

```python
from rank_bm25 import BM25Okapi

class HybridRetriever:
    def __init__(self, collection, chunks):
        self.collection = collection
        self.chunks = chunks
        
        # Build BM25 index for keyword search
        tokenized_docs = [chunk['text'].split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def retrieve(self, query, top_k=10, semantic_weight=0.7):
        """Hybrid retrieval combining semantic + keyword search"""
        
        # 1. Semantic search (ChromaDB)
        semantic_results = self.collection.query(
            query_texts=[query],
            n_results=top_k * 2  # Fetch more candidates
        )
        
        # 2. Keyword search (BM25)
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # 3. Combine scores
        final_scores = {}
        for chunk_id, semantic_score in zip(
            semantic_results['ids'][0],
            semantic_results['distances'][0]
        ):
            # Normalize semantic score (distance → similarity)
            sem_score = 1 / (1 + semantic_score)
            
            # Get BM25 score
            chunk_idx = self.get_chunk_index(chunk_id)
            bm25_score = bm25_scores[chunk_idx]
            
            # Weighted combination
            final_scores[chunk_id] = (
                semantic_weight * sem_score +
                (1 - semantic_weight) * bm25_score
            )
        
        # 4. Re-rank and return top-k
        ranked_chunks = sorted(
            final_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return [self.chunks[cid] for cid, _ in ranked_chunks]
```

### Query Expansion

**Expand user query with financial domain synonyms**

```python
def expand_query(query):
    """Expand query with domain-specific synonyms"""
    
    synonyms = {
        "revenue": ["sales", "net sales", "top line"],
        "profit": ["earnings", "net income", "bottom line"],
        "R&D": ["research and development", "innovation spending"],
        "YoY": ["year-over-year", "annual growth"],
    }
    
    expanded_terms = [query]
    for term, alternatives in synonyms.items():
        if term.lower() in query.lower():
            expanded_terms.extend(alternatives)
    
    return " OR ".join(expanded_terms)
```

### Re-ranking with LLM

**Use Groq LLM to re-rank retrieved chunks by relevance**

```python
from langchain_groq import ChatGroq

def rerank_with_llm(query, chunks, top_k=5):
    """Re-rank chunks using LLM relevance scoring"""
    
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    
    # Score each chunk
    scored_chunks = []
    for chunk in chunks:
        prompt = f"""
        Query: {query}
        
        Document Chunk:
        {chunk['text'][:500]}...
        
        Rate the relevance of this chunk to the query on a scale of 0-10.
        Respond with only a number.
        """
        
        score = float(llm.invoke(prompt).content.strip())
        scored_chunks.append((chunk, score))
    
    # Sort by score and return top-k
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in scored_chunks[:top_k]]
```

---

## Implementation Details

### File Structure

```
src/task1_chunking/
├── parsers/
│   ├── llamaparse_handler.py    # LlamaParse integration
│   ├── pymupdf_parser.py        # PyMuPDF fallback
│   ├── table_extractor.py       # Table extraction logic
│   └── figure_extractor.py      # Figure extraction logic
│
├── chunkers/
│   ├── semantic_chunker.py      # Semantic chunking
│   ├── structure_aware.py       # Structure-aware chunking
│   └── multimodal_chunker.py    # Multimodal integration
│
└── storage/
    ├── chromadb_manager.py      # ChromaDB operations
    ├── embedding_handler.py     # Groq embeddings
    └── retriever.py             # Hybrid retrieval
```

### Usage Example

```python
from src.task1_chunking import DocumentProcessor

# Initialize processor
processor = DocumentProcessor(
    parser="llamaparse",  # or "pymupdf"
    chunking_strategy="multimodal",
    chunk_size=1024,
    chunk_overlap=128
)

# Process document
chunks = processor.process_document(
    pdf_path="data/raw/Amazon/AMAZON_2022_10K.pdf",
    document_id="AMAZON_2022_10K"
)

# Index in ChromaDB
processor.index_chunks(chunks)

# Retrieve relevant chunks
results = processor.retrieve(
    query="What was Amazon's revenue growth in 2022?",
    top_k=5,
    filters={"year": 2022}
)
```

---

## Performance Metrics

**Chunking Quality Metrics**:
- **Context Preservation**: Measure of structural information retained
- **Retrieval Accuracy**: Precision@5, Recall@10 on test queries
- **Chunk Size Distribution**: Histogram of chunk sizes
- **Coverage**: Percentage of document content successfully chunked

**Storage Efficiency**:
- **Index Size**: Total ChromaDB storage size
- **Indexing Time**: Time to process and index one document
- **Query Latency**: Average retrieval time (semantic + hybrid)

**Evaluation Results** (see [EVALUATION.md](EVALUATION.md)):
- Semantic chunking: 0.87 retrieval precision
- Multimodal chunking: 0.92 precision (with table queries)
- Hybrid retrieval: 15% improvement over pure semantic search

---

## Next Steps

- Implement advanced table parsing with formula extraction
- Add support for charts/graphs analysis using vision models
- Experiment with recursive chunking for very long sections
- Optimize embedding caching for faster indexing

---

**Related Documentation**:
- [ARCHITECTURE.md](ARCHITECTURE.md) - System overview
- [TASK2_AGENTS.md](TASK2_AGENTS.md) - Multi-agent retrieval
- [SETUP.md](SETUP.md) - Setup instructions
