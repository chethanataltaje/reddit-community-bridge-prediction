# Cytoscape Network Import & Visualization Guide

This directory contains CSV files exported from the Reddit Community Bridge Prediction pipeline for visualization in [Cytoscape](https://cytoscape.org/).

---

## 1. File Descriptions

| File Name | Description | Key Columns |
|---|---|---|
| `nodes.csv` | Unique subreddits in the visualization subgraph | `id`, `subreddit`, `degree`, `community_id` |
| `edges.csv` | Existing training edges in `G_train` | `source`, `target`, `edge_type` (`existing_edge`) |
| `predicted_edges.csv` | Top-10 and Top-20 candidate link predictions | `source`, `target`, `algorithm`, `score`, `source_community`, `target_community`, `cross_community`, `rank` |

---

## 2. Subgraph Selection Rule

To ensure Cytoscape layouts remain clean, interactive, and readable:
* **Target Nodes:** All unique subreddit nodes appearing in the Top-20 predictions across Common Neighbors, Jaccard, Adamic-Adar, and Preferential Attachment are included.
* **Neighborhood Induced Subgraph:** The 1-hop neighbor subreddits in `G_train` connected to these target nodes (filtering out massive hub subreddits with degree $> 100$) are included along with all existing training edges between them.

---

## 3. Cytoscape Step-by-Step Import Instructions

### Step 1: Import Existing Network (`edges.csv`)
1. Open Cytoscape.
2. Go to **File $\rightarrow$ Import $\rightarrow$ Network from File...**
3. Select `cytoscape/edges.csv`.
4. In the import dialog, map:
   * `source` $\rightarrow$ **Source Node**
   * `target` $\rightarrow$ **Target Node**
   * `edge_type` $\rightarrow$ **Edge Attribute**
5. Click **OK** to build the base network topology.

### Step 2: Import Node Metadata (`nodes.csv`)
1. Go to **File $\rightarrow$ Import $\rightarrow$ Table from File...**
2. Select `cytoscape/nodes.csv`.
3. Set **Where to Import Table Data** to **To Selected Networks Only**.
4. Map `id` to **Key** (matching the node name/ID in Cytoscape).
5. Click **OK**. Node attributes `degree` and `community_id` are now loaded into the Node Table.

### Step 3: Overlay Predicted Edges (`predicted_edges.csv`)
1. Go to **File $\rightarrow$ Import $\rightarrow$ Network from File...**
2. Select `cytoscape/predicted_edges.csv`.
3. In the import dialog:
   * Select **To Existing Network** (or import as an overlay edge table).
   * Map `source` $\rightarrow$ **Source Node**
   * `target` $\rightarrow$ **Target Node**
   * Map `algorithm`, `score`, `cross_community`, `rank` $\rightarrow$ **Edge Attributes**.
4. Click **OK**.

---

## 4. Recommended Visual Style Mappings

### Node Styling
* **Node Color (Fill Color):** Map `community_id` using a **Discrete Mapping** (assign distinct colors to different Louvain community IDs).
* **Node Size:** Map `degree` using a **Continuous Mapping** (larger nodes represent higher-degree subreddits).
* **Node Label:** Map `subreddit`.

### Edge Styling
* **Existing Edges (`edge_type == 'existing_edge'`):**
  * Line Style: **Solid**
  * Stroke Color: Light Gray (`#CCCCCC`) or Muted Blue
  * Transparency/Opacity: 40%
* **Predicted Edges (`predicted_edges.csv`):**
  * **Cross-Community Candidates (`cross_community == True`):**
    * Line Style: **Dashed** or **Dotted**
    * Stroke Color: Bright Red (`#E74C3C`) or Bright Orange (`#E67E22`)
    * Line Width: Thicker (2–3 px) to highlight potential emerging bridges.
  * **Within-Community Candidates (`cross_community == False`):**
    * Line Style: **Solid**
    * Stroke Color: Dark Blue (`#2980B9`) or Green (`#27AE60`)
    * Line Width: Standard (1.5 px).

---

## 5. Methodological Caution

These visual overlays display high-scoring topological prediction candidates from the held-out candidate set. Cytoscape visual representations demonstrate structural positioning and community boundaries; they do NOT prove or guarantee that a hyperlink will form in the future.
