<p align="center">
  <h1 align="center">A multi-view entropy bottleneck contrastive learning based cancer early detection method using multi-omics data(MEBCL)</h1>
Alternative splicing (AS) enables a single gene to generate multiple transcript and protein isoforms, providing fine-grained molecular signals that are closely associated with early tumorigenesis and disease progression. Despite the growing adoption of multi-omics data for cancer early detection and risk prediction, AS-derived features remain largely underexplored in existing predictive frameworks.Here, we propose MEBCL, a multi-view entropy bottleneck contrastive learning–based model for cancer early-stage prediction that integrates copy number variation, DNA methylation, gene expression, miRNA expression, and alternative splicing data. MEBCL leverages self-supervised contrastive learning with redundancy reduction and entropy bottleneck constraints to learn robust and complementary representations across heterogeneous omics views, enabling accurate discrimination between normal and early-stage cancer samples.Comprehensive ablation experiments demonstrate that incorporating AS information consistently improves early prediction performance across multiple cancer types, highlighting the unique value of splicing-level signals beyond conventional expression-based features. Furthermore, we identify a set of early cancer–associated AS events and conduct systematic downstream analyses, including open reading frame annotation, RNA-binding protein–mediated splicing regulation, and splicing-derived anticancer peptide (ACP) prediction, to explore their potential biological and therapeutic relevance.
<br>
<br>
<p align="center">
<img width="800" height="466" alt="image" src="https://github.com/user-attachments/assets/65865e0a-d386-42ee-a0b2-eb436e10ffb4" />
<br>
<br>
<br>

## Dependencies

MEBCL is an early cancer prediction framework based on multi-view entropy bottleneck contrastive learning. It performs binary classification by taking patient-level multi-omics profiles as input and outputting the early-stage cancer risk (Normal vs Early). The method first learns robust and complementary representations across heterogeneous omics views via self-supervised contrastive pretraining with an entropy bottleneck and redundancy reduction constraints. The pretrained representations are then fine-tuned for early cancer prediction. To improve stability and prevent representation collapse without requiring negative pairs or asymmetric architectures, MEBCL employs a dedicated contrastive objective combined with entropy bottleneck regularization.

The model is developed with Python 3.8.20, PyTorch 2.2.0, and CUDA 11.8. For other environments, please refer to environment.yml.

## Installation

1.Download and install Miniconda.

2.Create a conda environment and install required packages:
```bash
conda create -n mebcl && conda activate mebcl
bash environment.sh
```
The environment can also be created by conda env `create -f environment.yml`

## Start

### Early Cancer Prediction

Example data (e.g., BRCA) is provided in the `test/` directory and can be downloaded for quick testing. If you want to run MEBCL on your own data, please refer to the data preprocessing procedure described in the paper to prepare multi-omics inputs.

Before running the model, please modify the configuration file if necessary.  
For testing with the provided example data, the default configuration can be used directly.  
For custom datasets, please update the cancer type name and related settings in the corresponding `.yaml` configuration file.

Run the following command:

```bash
python main.py
```
#### Results include：
MEBCL outputs a binary prediction indicating Normal or Early-stage cancer for each sample.
The performance of the model is evaluated using standard classification metrics, including:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- AUC

<p align="center">
<img width="419" height="347" alt="image" src="https://github.com/user-attachments/assets/1b6ac280-cf5b-40c4-af9e-b4a441a47b3a" />
<img  width="419" height="279" alt="image" src="https://github.com/user-attachments/assets/49b5a370-be36-4bbf-966f-1761de144d44" />
</p>

Extensive experiments demonstrate that MEBCL consistently outperforms baseline multi-omics methods across multiple cancer types. Ablation studies further indicate that incorporating alternative splicing features significantly improves early cancer prediction performance.

Detailed quantitative results and visualizations are provided in the out/ directory.
