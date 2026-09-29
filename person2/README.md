# Person 2 Implementation Package: Community Bridge Evaluation & Analysis

This package contains Person 2's implementation for evaluating link prediction algorithms, detecting Louvain community structures, analyzing cross-community bridge predictions, generating Cytoscape network exports, and rendering result visualizations.

---

## Directory & File Structure

```
person2/
├── evaluation/
│   ├── __init__.py
│   └── evaluation.py                  # Evaluation metrics (Precision@K, Recall@K, ROC-AUC, threshold metrics)
├── predictions/
│   ├── __init__.py
│   ├── predictions.py                 # Top-K candidate link extraction and deduplication
│   ├── top10_predictions.csv          # Top-10 predictions table
│   └── top20_predictions.csv          # Top-20 predictions table
├── community/
│   ├── __init__.py
│   ├── community_detection.py         # Louvain community detection on G_train
│   ├── bridge_analysis.py             # Cross-community candidate annotation
│   ├── community_results.csv          # Subreddit to community ID mapping
│   ├── bridge_analysis.csv            # Annotated candidate predictions
│   └── bridge_summary.csv             # Top-K cross-community summary table
├── cytoscape/
│   ├── README.md                      # Cytoscape import guide and styling rules
│   ├── nodes.csv                      # Node metadata table for Cytoscape
│   ├── edges.csv                      # Existing training edges for Cytoscape
│   └── predicted_edges.csv            # Candidate predictions overlay for Cytoscape
├── visualizations/
│   ├── generate_plots.py              # Matplotlib visualization plotting functions
│   ├── algorithm_precision_roc_comparison.png
│   ├── algorithm_recall_comparison.png
│   ├── cross_community_bridge_proportion.png
│   ├── community_size_distribution.png
│   └── roc_curves.png
├── tests/
│   ├── test_evaluation.py             # Unit tests for evaluation module
│   ├── test_predictions.py            # Unit tests for predictions module
│   ├── test_community.py              # Unit tests for community module
│   └── test_cytoscape_visualizations.py # Unit tests for Cytoscape & visualizations
├── run_pipeline_evaluation.py        # Pipeline runner for evaluation metrics
├── run_pipeline_predictions.py       # Pipeline runner for Top-K extraction
├── run_community_analysis.py         # Pipeline runner for community detection & bridge analysis
├── run_visualization_pipeline.py     # Pipeline runner for Cytoscape export & figures
└── evaluation_results.csv            # Authoritative comparison results CSV table
```

---

## Unit Test Suite Execution
To run all 25 unit tests across the 4 test modules:
```bash
python person2/tests/test_evaluation.py
python person2/tests/test_predictions.py
python person2/tests/test_community.py
python person2/tests/test_cytoscape_visualizations.py
```
All 25 unit tests pass cleanly.
