# Predicting Emerging Community Bridges in Reddit Using Social Network Analysis and Link Prediction

An end-to-end Social Network Analysis (SNA) and Link Prediction framework to detect, evaluate, and analyze potential emerging community bridges in Reddit using topological heuristics and Louvain community detection.

---

## 1. Project Overview & Problem Definition

Digital social networks like Reddit consist of interconnected communities (subreddits) that continuously evolve and interact via cross-hyperlinks. Understanding how and why distinct online communities interact is essential for modeling information diffusion, tracking platform polarization, and detecting echo chambers.

This project formulates **Community Bridge Prediction** as a topological Link Prediction problem:
Given a static snapshot of subreddit interactions, can we predict which communities are most likely to form *new* hyperlinks with each other in the future, relying purely on the underlying topological structure of the network?

---

## 2. Alignment with UN Sustainable Development Goals (SDGs)

* **SDG 9 (Industry, Innovation and Infrastructure):** Developing advanced network analysis models and predictive link heuristics improves the algorithmic infrastructure and information routing efficiency of modern digital communication platforms.
* **SDG 16 (Peace, Justice and Strong Institutions):** Mapping and predicting cross-community interactions empowers platform researchers to monitor digital polarization, track the propagation of misinformation across community boundaries, and support healthier online discourse institutions.

---

## 3. Project Directory Structure

```
reddit-community-bridge-prediction/
├── .gitignore                         # Environment & data ignore configuration
├── README.md                          # Comprehensive project documentation
├── requirements.txt                   # Environment package dependencies
│
├── notebooks/
│   └── 01_reddit_community_bridge_prediction.ipynb # Complete integrated Jupyter notebook
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py          # Data loading, validation, and serialization
│   ├── network_analysis.py            # Graph construction & exploratory SNA metrics
│   ├── link_prediction.py             # Topological heuristics (CN, Jaccard, AA, PA)
│   ├── evaluation.py                  # Evaluation metrics (Precision@K, Recall@K, ROC-AUC)
│   ├── predictions.py                 # Top-K candidate extraction & deduplication
│   ├── community_detection.py         # Louvain community detection on G_train
│   ├── bridge_analysis.py             # Cross-community candidate annotation
│   ├── visualization.py               # Publication-quality Matplotlib charts
│   └── cytoscape_export.py            # Network & predicted edge CSV exports for Cytoscape
│
├── results/
│   ├── evaluation_results.csv         # Authoritative metrics comparison table
│   ├── top10_predictions.csv          # Top-10 predictions across heuristics
│   ├── top20_predictions.csv          # Top-20 predictions across heuristics
│   ├── community_results.csv          # Subreddit-to-community mapping (54,075 nodes)
│   ├── bridge_analysis.csv            # Annotated candidate predictions with community IDs
│   └── bridge_summary.csv             # Top-K cross-community proportion summary
│
├── visualizations/
│   ├── algorithm_precision_roc_comparison.png
│   ├── algorithm_recall_comparison.png
│   ├── cross_community_bridge_proportion.png
│   ├── community_size_distribution.png
│   └── roc_curves.png
│
├── cytoscape/
│   ├── README.md                      # Cytoscape import guide and styling rules
│   ├── nodes.csv                      # Node metadata table for Cytoscape
│   ├── edges.csv                      # Existing training edges for Cytoscape
│   └── predicted_edges.csv            # Candidate predictions overlay for Cytoscape
│
└── tests/
    ├── test_evaluation.py             # Unit tests for evaluation module
    ├── test_predictions.py            # Unit tests for predictions module
    ├── test_community.py              # Unit tests for community module
    └── test_visualizations.py         # Unit tests for Cytoscape & visualizations
```

---

## 4. Methodology & Pipeline

1. **Dataset Preprocessing:** Stanford SNAP `soc-RedditHyperlinks-title.tsv` (55,863 subreddits, 858,490 raw hyperlinks). Self-loops are removed, and directed multi-edges are collapsed into a clean, undirected, unweighted simple graph $G$.
2. **Exploratory SNA:** Macro topological metrics are calculated ($N = 54,075$ nodes, $E = 220,151$ unique edges, average degree $\approx 8.14$, density $\approx 0.00015$).
3. **Train/Test Edge Split:** Random 80/20 edge split yields training graph $G_{train}$ ($176,121$ edges) and held-out positive test edges ($44,030$ edges).
4. **Negative Sampling:** Uniform random sampling of $44,030$ non-edges guarantees a balanced 1:1 candidate test set ($88,060$ total candidate edges).
5. **Louvain Community Detection:** Applied strictly to $G_{train}$ (seed=42) to partition nodes into $6,795$ modular communities without data leakage.
6. **Heuristic Scoring:** Scores candidate edges using Common Neighbors (CN), Jaccard Coefficient (JC), Adamic-Adar Index (AA), and Preferential Attachment (PA).
7. **Evaluation:** Computes scale-invariant ranking metrics (`Precision@10`, `Precision@20`, `ROC-AUC`), rank-based `Recall@K`, and threshold-based classification metrics (`Recall@0.5`, `F1@0.5`).
8. **Bridge Analysis:** Identifies cross-community candidate link predictions (`source_community != target_community`) and computes Top-K bridge proportions.

---

## 5. Summary Evaluation Results

| Algorithm | Precision@10 | Precision@20 | Recall@10 | Recall@20 | Recall@0.5 | F1@0.5 | ROC-AUC |
|---|---|---|---|---|---|---|---|
| **Common Neighbors** | 1.0000 | 1.0000 | 0.000227 | 0.000454 | 0.730706 | 0.834405 | 0.860476 |
| **Jaccard** | 1.0000 | 0.9500 | 0.000227 | 0.000432 | 0.001635 | 0.003262 | 0.853955 |
| **Adamic-Adar** | 1.0000 | 1.0000 | 0.000227 | 0.000454 | 0.532569 | 0.694333 | **0.861245** |
| **Preferential Attachment** | 1.0000 | 1.0000 | 0.000227 | 0.000454 | **0.860005** | 0.648879 | 0.857944 |

---

## 6. Key Findings

* **Predictive Heuristics:** Adamic-Adar achieves the highest overall ROC-AUC ($0.8612$) by penalizing high-degree hubs, reducing false-positive bridge recommendations.
* **Cross-Community vs Intra-Community Behavior:** Top candidate predictions from Common Neighbors ($85\%-100\%$), Adamic-Adar ($85\%-90\%$), and Preferential Attachment ($90\%-100\%$) are **predominantly cross-community** (potential emerging structural bridges). Conversely, Jaccard predictions ($0\%$ cross-community) are **strictly intra-community** (niche, highly overlapping local subreddits).
* **Candidate Link Status:** High-scoring predictions represent candidate topological connections derived from structural proximity in $G_{train}$; they indicate potential emerging links rather than guaranteed future interactions.

---

## 7. How to Run

### Installation
```bash
pip install -r requirements.txt
```

### Running Unit Tests
```bash
python -m unittest discover -s tests
```

### Running the Integrated Notebook
```bash
python -m nbconvert --to notebook --execute --inplace notebooks/01_reddit_community_bridge_prediction.ipynb
```
