# Semantic text baseline smoke results

- Frozen semantic-fusion BR w=1 C=1 (mature_factual 2024): Micro-F1=0.651, Macro-F1=0.427, Samples-F1=0.648, Jaccard=0.534
- BR-TFIDF same split (mature_factual 2024): Micro-F1=0.645, Macro-F1=0.417, Samples-F1=0.641, Jaccard=0.523
- BR-TFIDF same split (mature_factual 2023): Micro-F1=0.641, Macro-F1=0.405, Samples-F1=0.632, Jaccard=0.514
- Frozen semantic-fusion BR w=1 C=1 (mature_factual 2023): Micro-F1=0.632, Macro-F1=0.401, Samples-F1=0.624, Jaccard=0.504
- BR-TFIDF same split (preliminary 2023): Micro-F1=0.551, Macro-F1=0.336, Samples-F1=0.541, Jaccard=0.403
- Frozen semantic-fusion BR w=1 C=1 (preliminary 2023): Micro-F1=0.546, Macro-F1=0.326, Samples-F1=0.535, Jaccard=0.397
- Frozen semantic-fusion BR w=1 C=1 (preliminary 2024): Micro-F1=0.539, Macro-F1=0.321, Samples-F1=0.532, Jaccard=0.398
- Segmented MiniLM-BR C=1 (mature_factual 2024): Micro-F1=0.534, Macro-F1=0.329, Samples-F1=0.543, Jaccard=0.413
- BR-TFIDF same split (preliminary 2024): Micro-F1=0.521, Macro-F1=0.325, Samples-F1=0.516, Jaccard=0.381
- Segmented MiniLM-BR C=1 (mature_factual 2023): Micro-F1=0.506, Macro-F1=0.325, Samples-F1=0.512, Jaccard=0.381
- Segmented MiniLM-BR C=1 (preliminary 2023): Micro-F1=0.463, Macro-F1=0.279, Samples-F1=0.456, Jaccard=0.315
- Segmented MiniLM-BR C=1 (preliminary 2024): Micro-F1=0.457, Macro-F1=0.272, Samples-F1=0.448, Jaccard=0.310

Best semantic method change against BR-TFIDF: +0.100 Micro-F1.