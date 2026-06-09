# Group Evaluation Results

## Config A

- Backend: `openrouter`

| Metric | Score |
|---|---:|
| citation_coverage | 1.000 |
| answer_relevance | 0.300 |
| context_recall | 0.300 |
| context_precision | 0.120 |

## Config B

- Backend: `hybrid_no_rerank`

| Metric | Score |
|---|---:|
| citation_coverage | 1.000 |
| answer_relevance | 0.356 |
| context_recall | 0.356 |
| context_precision | 0.147 |

## Notes

- Config A uses the full retrieval pipeline with reranking.
- Config B uses dense + lexical merge without reranking.
- Winner on this run: `Config B`.