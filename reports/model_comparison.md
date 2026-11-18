# SmartRecommend — Résultats Expérimentaux sur le Test Set Temporel

| Modèle | HitRate@10 | NDCG@10 | MAP@10 | Recall@10 | Precision@10 | Diversity@10 | Coverage | Latence (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **1. Popularity Baseline** | 0.0100 | 0.0042 | 0.0026 | 0.0090 | 0.0010 | 0.9006 | 0.0001 | 0.13 |
| **2. Content-Based** | 0.1040 | 0.0677 | 0.0583 | 0.0941 | 0.0106 | 0.1018 | 0.0206 | 126.25 |
| **3. Collaborative ALS** | 0.0780 | 0.0430 | 0.0345 | 0.0611 | 0.0100 | 0.4960 | 0.0051 | 5.34 |
| **4. Session-Based** | 0.1040 | 0.0575 | 0.0471 | 0.0735 | 0.0118 | 0.2727 | 0.0062 | 0.12 |
| **5. Hybrid Weighted** | 0.1660 | 0.0878 | 0.0674 | 0.1315 | 0.0194 | 0.3360 | 0.0141 | 146.17 |
| **6. Hybrid LightGBM LTR** | 0.1720 | 0.0961 | 0.0758 | 0.1407 | 0.0196 | 0.3416 | 0.0153 | 306.99 |
