# LCG-HGNN: Logic-Constrained Gene-Pathway Heterogeneous Graph Neural Network

**Inferring signaling pathway abnormalities from histopathological images via logic-constrained gene-pathway heterogeneous knowledge graph**

![Status](https://img.shields.io/badge/Status-Published-success) ![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D1.12-red) ![License](https://img.shields.io/badge/License-MIT-blue)

This repository contains the official implementation of **LCG-HGNN**, a framework designed to infer signaling pathway alterations (e.g., PI3K-Akt) directly from Whole-Slide Images (WSIs). By integrating a **Gene-Pathway Heterogeneous Graph** and the **KePathGraph** framework, this model enables collaborative recognition of gene groups and incorporates biological priors via logical clauses to enhance clinical interpretability.

This codebase provides a complete, self-contained pipeline for data preprocessing, model training, and evaluation, facilitating the reproduction of the experimental results reported in our paper.

## Table of Contents

- [System Requirements](#-system-requirements)
- [Installation](#-installation)
- [Dataset Preparation](#-dataset-preparation)
- [Hyperparameters & Configuration](#-hyperparameters--configuration)
- [Usage Pipeline (Reproduction Steps)](#-usage-pipeline-reproduction-steps)
  - [Step 1: Graph Learning & Dynamic Weighting](#step-1-graph-learning--dynamic-weighting)
  - [Step 2: Node Sorting & Selection](#step-2-node-sorting--selection)
  - [Step 3: Feature Embedding](#step-3-feature-embedding)
  - [Step 4: KePathGraph Construction & Knowledge Enhancement](#step-4-kepathgraph-construction--knowledge-enhancement)
- [Inputs and Outputs](#-inputs-and-outputs)
- [Citation](#-citation)

## System Requirements

* **OS:** Linux (Recommended) or Windows
* **Python:** 3.8+
* **Hardware:** NVIDIA GPU (Tested on RTX A6000) with CUDA 11.x+
* **Dependencies:**
  * `torch >= 1.12`
  * `torch-geometric` (Crucial for GIN/GCN layers)
  * `torchvision`
  * `scikit-learn`, `pandas`, `numpy`, `pillow`
  * `KENN-PyTorch` (See installation below)

## Installation

To ensure reproducibility, please follow these steps to set up the environment and install the required Knowledge Enhanced Neural Networks (KENN) library.

1. **Clone the repository:**

   ```bash
   git clone [https://github.com/xianyvxxx/LCG-HGNN.git](https://github.com/xianyvxxx/LCG-HGNN.git)
   cd LCG-HGNN
   ```

2. **Create a virtual environment:**

   ```bash
   conda create -n lcg_hgnn python=3.8
   conda activate lcg_hgnn
   ```

3. **Install PyTorch (adjust for your CUDA version):**

   ```bash
   pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url [https://download.pytorch.org/whl/cu113](https://download.pytorch.org/whl/cu113)
   ```

4. **Install PyTorch Geometric:**

   ```bash
   pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f [https://data.pyg.org/whl/torch-1.12.1+cu113.html](https://data.pyg.org/whl/torch-1.12.1+cu113.html)
   pip install pandas numpy scikit-learn pillow
   ```

5. **Install KENN (Crucial Step):**
   We utilize **KENN-PyTorch** for logic constraint layers. Please install it directly from the official source:

   ```bash
   # Clone and install KENN
   git clone [https://github.com/HEmile/KENN-PyTorch.git](https://github.com/HEmile/KENN-PyTorch.git)
   cd KENN-PyTorch
   pip install .
   cd ..
   ```

## Dataset Preparation

To reproduce the results, the data must be organized and pre-processed according to the methods described in our manuscript.

### 1. Primary Data Sources

| Dataset                       | Description                                                  | Access                                                       | Usage in Study                                               |
| ----------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **TCGA-LUAD WSIs**            | 1,608 whole-slide images (`.svs` format) from lung adenocarcinoma patients | [NCI Genomic Data Commons (GDC)](https://portal.gdc.cancer.gov/projects/TCGA-LUAD) | Histopathological image input; linked to genomic data via TCGA barcode |
| **cBioPortal Mutation Data**  | Somatic mutation profiles (SNVs/indels) for TCGA-LUAD cohort (585 patients with WSI + mutation) | [cBioPortal API](https://www.cbioportal.org/study/summary?id=luad_tcga) | Gene-level labels for model supervision                      |
| **Cancer Gene Census (CGC)**  | Curated list of 35 high-confidence lung cancer driver genes (e.g., *EGFR*, *KRAS*, *TP53*) with oncogenic/tumor-suppressor roles | [COSMIC CGC v102](https://cancer.sanger.ac.uk/census)        | Prior knowledge for edge weighting & label filtering         |
| **DAVID Pathway Annotations** | Gene–pathway associations from 4 databases: KEGG, Reactome, WikiPathways, BioCarta | [DAVID 6.8 API](https://david.ncifcrf.gov/)                  | Construction of gene–pathway heterogeneous graph             |
| **Human Protein Atlas (HPA)** | Immunohistochemistry images of *EGFR*/KRAS*-mutant LUAD cases (clinical reference) | [HPA Pathology Atlas](https://www.proteinatlas.org/pathology) | Validation of clinical plausibility (image similarity assessment) |

### 2. Directory Structure

We recommend organizing your data directory (`/your/data/root/`) as follows. **Note:** You must update the path variables in the scripts (`GIN_TaT.py`, etc.) to point to your specific directories.

```text
data_root/
├── label/
│   ├── gene_list.csv          # Gene labels (Target)
│   ├── pathway_new.csv        # Pathway hierarchy info (DAVID/KEGG derived)
│   └── top20_LUAD.csv         # Patient IDs and mutation status
├── patch/
│   ├── features/              # Pre-processed .json graph features (See Logic below)
│   └── images/                # Raw WSI patches (Required for Step 3 visualization)
├── logic_rules/
│   └── logic_rules_hsa04151.txt # DNF Logic clauses for the pathway
└── output/                    # Directory for saved models and results
```

### 3. Preprocessing Logic 

Before running the pipeline, raw WSIs must be processed into graph features:

1. **Segmentation:** Segment WSIs into $512 \times 512$ pixel patches.
2. **Feature Extraction:** Use an ImageNet-pretrained **ResNet-18** to extract features for each patch.
3. **Graph Construction:**
   - Compute Euclidean distances between patch features.
   - Connect each node to its top-10 nearest neighbors.
   - Establish spatial edges if physical distance is < 85 pixels.
4. **Format:** Save the resulting adjacency matrices (`adj`) and feature vectors (`x`) into `.json` format compatible with `generate_dataset` in `functions/utils.py`.

## Hyperparameters & Configuration

We provide configuration with default hyperparameters that reproduce the main experimental results reported in the paper. These are located at the top of `GIN_TaT.py` and `ke_graph_s.py`.

**Key Default Settings:**

- `feature_dimension`: 512
- `num_epochs`: 128
- `batch_size`: 24
- `learning_rate`: 1e-3
- `weight_decay`: 1e-5
- `lambda_reg`: 0.03 (For dynamic edge weighting regularization)

## Usage Pipeline (Reproduction Steps)

The framework operates in a sequential pipeline. Follow these steps to reproduce the main experimental results.

### Step 1: Graph Learning & Dynamic Weighting

Train the **Gene-Pathway Heterogeneous Graph** model. This script trains the GIN backbone to predict gene mutations and applies the **Dynamic Edge Weighting Mechanism** to incorporate **Cancer Gene Census (CGC)** priors.

Bash

```
# Update 'csv_file_gene' and 'feature_path' in GIN_TaT.py first
python GNN_train_and_test.py
```

- **Function:** Trains the model using multi-label graph classification.

- **Key Logic:** Implements the dynamic weighting formula:

  $$\Delta w_{gp}=\gamma_{gp}\cdot e\cdot w_{gp}-\lambda\cdot\frac{w_{gp}}{|w_{gp}|_{2}}$$

- **Expected Output:** Trained model checkpoints (e.g., `epoch_97.pth`) saved in the output directory and training logs showing Accuracy/AUC metrics.

### Step 2: Node Sorting & Selection

Run inference on the test set to identify the "Top-K" most structurally important patches (Global Sort Pooling).

Bash

```
# Update 'model_path' to point to the best checkpoint from Step 1
python ke_data.py
```

- **Function:** Loads the best checkpoint and extracts patch indices contributing to the prediction.
- **Expected Output:** `test_data_patch.pth` (list of important patch paths) and `select_index_all.pth`.

### Step 3: Feature Embedding

Extract visual feature embeddings for the specific patches identified in Step 2. These embeddings are required for building the inter-individual graph.

Bash

```
python ke_embeding.py
```

- **Function:** Uses ResNet-18 to generate tensor embeddings for the selected patches.
- **Expected Output:** A directory `result_dict/` containing `.pt` embedding files for the top-ranked patches.

### Step 4: KePathGraph Construction & Knowledge Enhancement

Construct the **KePathGraph** and apply **Logical Clauses** (e.g., Mutual Exclusivity, Hierarchy). This step refines the predictions using the **Knowledge Enhancer (KE)** module from KENN.

Bash

```
# Ensure 'logic_rules_hsa04151.txt' is available
python ke_graph_s.py
```

- **Function:** Builds the inter-individual graph and applies logic rules (Equations 4-9 in the manuscript).
- **Expected Output:**
  - `mismatched_rows.csv`: Analysis of predictions refined by logic constraints.
  - Comparative metrics showing performance with vs. without logic constraints.

## Inputs and Outputs Summary

| **Script**              | **Input Data**                       | **Output Data**                           |
| ----------------------- | ------------------------------------ | ----------------------------------------- |
| `GNN_train_and_test.py` | Graph feature JSONs, Label CSVs      | Model checkpoints (`.pth`), Training logs |
| `ke_data.py`            | Trained Model (`.pth`), Test Dataset | Selected patch indices (`.pth`)           |
| `ke_embeding.py`        | Selected patch images                | Visual embeddings (`.pt`)                 |
| `ke_graph_s.py`         | Embeddings, Logic Rules (`.txt`)     | Final Predictions, Consistency CSVs       |

## Citation

If you find this code or framework useful for your research, please cite our paper:

```
@article{LCG-HGNN,
  title={Inferring signaling pathway abnormalities from histopathological images via logic-constrained gene-pathway heterogeneous knowledge graph},
  author={Yu, Yu and Shi, Wen and Chen, Xin and Feng, Jinghui and Huang, Simin and Zeng, Shixian and Bo, Xiaolin and Xi, Jianing},
  journal={npj Biomedical Innovations},
  year={2025}
}
```

