# Semantic text baseline smoke results

- Frozen semantic-fusion BR w=1 C=1 (mature_factual 2024): Micro-F1=0.686, Macro-F1=0.468, Samples-F1=0.677, Jaccard=0.573
- BR-TFIDF check (mature_factual 2024): Micro-F1=0.674, Macro-F1=0.473, Samples-F1=0.668, Jaccard=0.556
- BR-TFIDF check (mature_factual 2023): Micro-F1=0.673, Macro-F1=0.440, Samples-F1=0.664, Jaccard=0.557
- Frozen semantic-fusion BR w=1 C=1 (mature_factual 2023): Micro-F1=0.663, Macro-F1=0.441, Samples-F1=0.659, Jaccard=0.545
- BR-TFIDF check (preliminary 2023): Micro-F1=0.579, Macro-F1=0.353, Samples-F1=0.567, Jaccard=0.437
- Frozen semantic-fusion BR w=1 C=1 (preliminary 2023): Micro-F1=0.563, Macro-F1=0.345, Samples-F1=0.552, Jaccard=0.419
- BR-TFIDF check (preliminary 2024): Micro-F1=0.562, Macro-F1=0.334, Samples-F1=0.548, Jaccard=0.422
- Frozen semantic-fusion BR w=1 C=1 (preliminary 2024): Micro-F1=0.555, Macro-F1=0.340, Samples-F1=0.544, Jaccard=0.417
- Segmented MiniLM-BR C=1 (mature_factual 2024): Micro-F1=0.548, Macro-F1=0.333, Samples-F1=0.554, Jaccard=0.438
- Segmented MiniLM-BR C=1 (mature_factual 2023): Micro-F1=0.539, Macro-F1=0.324, Samples-F1=0.542, Jaccard=0.422
- Segmented MiniLM-BR C=1 (preliminary 2023): Micro-F1=0.474, Macro-F1=0.277, Samples-F1=0.470, Jaccard=0.339
- Segmented MiniLM-BR C=1 (preliminary 2024): Micro-F1=0.453, Macro-F1=0.267, Samples-F1=0.448, Jaccard=0.317

Best semantic method change against BR-TFIDF: +0.108 Micro-F1.