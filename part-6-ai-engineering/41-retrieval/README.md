# Module 41: Retrieval

*Part VI: AI Engineering · about 35 hours*

Giving the model the right context at the right time. Harder than it sounds.

## Self-check

If you can answer these without looking anything up, skim the lessons and go straight to the labs.
If you can't, good, that's what the module is for.

- Why does pure vector search fail on product codes and error IDs?

## When you finish it, you can

- Use embeddings for semantic search and pick an embedding model with evidence.
- Explain and tune approximate nearest neighbor indexes (HNSW, IVF, PQ).
- Build a RAG pipeline with chunking, hybrid search (BM25 + dense), reranking and citations.
- Evaluate retrieval (recall@k, MRR, nDCG) and generation (faithfulness), and use advanced patterns (query rewriting, HyDE, GraphRAG, agentic retrieval).

## Lessons

41.1. [Embeddings and semantic search](41.1-embeddings-semantic-search.md)

41.2. [Vector indexes: HNSW, IVF, PQ](41.2-vector-indexes.md)

41.3. [RAG done properly](41.3-rag.md)

41.4. [Advanced RAG and evaluating retrieval](41.4-advanced-rag-evaluation.md)

## Labs

Run them from the repository root, for example:

```bash
python part-6-ai-engineering/41-retrieval/labs/lab_41_1_embeddings_search.py
```

- [`lab_41_1_embeddings_search.py`](labs/lab_41_1_embeddings_search.py)
- [`lab_41_2_vector_indexes.py`](labs/lab_41_2_vector_indexes.py)
- [`lab_41_3_rag.py`](labs/lab_41_3_rag.py)
- [`lab_41_4_advanced_rag_eval.py`](labs/lab_41_4_advanced_rag_eval.py)

Back to [Part VI](../) · [Syllabus](../../SYLLABUS.md)
