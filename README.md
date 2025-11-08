# LCG-HGNN

 **Inferring signaling pathway abnormalities directly from histopathological whole-slide images—without genomic sequencing.**  

Primary Data Sources

| Dataset | Description | Access | Usage in Study |
|--------|-------------|--------|----------------|
| **TCGA-LUAD WSIs** | 1,608 whole-slide images (`.svs` format) from lung adenocarcinoma patients | [NCI Genomic Data Commons (GDC)](https://portal.gdc.cancer.gov/projects/TCGA-LUAD) | Histopathological image input; linked to genomic data via TCGA barcode |
| **cBioPortal Mutation Data** | Somatic mutation profiles (SNVs/indels) for TCGA-LUAD cohort (585 patients with WSI + mutation) | [cBioPortal API](https://www.cbioportal.org/study/summary?id=luad_tcga) | Gene-level labels for model supervision |
| **Cancer Gene Census (CGC)** | Curated list of 35 high-confidence lung cancer driver genes (e.g., *EGFR*, *KRAS*, *TP53*) with oncogenic/tumor-suppressor roles | [COSMIC CGC v102](https://cancer.sanger.ac.uk/census) | Prior knowledge for edge weighting & label filtering |
| **DAVID Pathway Annotations** | Gene–pathway associations from 4 databases: KEGG, Reactome, WikiPathways, BioCarta | [DAVID 6.8 API](https://david.ncifcrf.gov/) | Construction of gene–pathway heterogeneous graph |
| **Human Protein Atlas (HPA)** | Immunohistochemistry images of *EGFR*/KRAS*-mutant LUAD cases (clinical reference) | [HPA Pathology Atlas](https://www.proteinatlas.org/pathology) | Validation of clinical plausibility (image similarity assessment) |
