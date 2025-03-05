# Part 2: Coding Challenge - Multimodal Gene Expression Analysis Project

## Project Overview

> **Note:** This page is just a high-level report of the part 2 results, plots and system components. For a more detailed explanation and running experiments please review [part2.ipynb](https://github.com/pouyan9675/ubiquitous-octo-eureka/blob/0630b9fb4346fa2d411b7c44ad5299c1d4a4a6c9/part-2/part2.ipynb).

This project implements a multimodal learning system that integrates textual embeddings from cell type labels with single-cell transcriptome embeddings to predict donor identifiers. The system uses data from the CellxGene Census platform, specifically focusing on human RNA measurements from the central nervous system.

The primary goal is to evaluate whether single-cell gene expression data, when combined with cell type label information, can effectively predict donor identifiers. To control for potential confounding variables, the system incorporates adversarial training to mitigate the effect of sex as a confounding factor.


## Key Components

The project is organized in a modular fashion with the following key components:

### Data Module (`utils/data.py`)
- [🔗](https://github.com/pouyan9675/ubiquitous-octo-eureka/blob/0630b9fb4346fa2d411b7c44ad5299c1d4a4a6c9/part-2/utils/data.py#L15) `CellDataset`: A custom PyTorch dataset for handling cell data
- [🔗](https://github.com/pouyan9675/ubiquitous-octo-eureka/blob/0630b9fb4346fa2d411b7c44ad5299c1d4a4a6c9/part-2/utils/data.py#L59) `load_ann_data`: Function to load AnnData from CellxGene Census
- [🔗](https://github.com/pouyan9675/ubiquitous-octo-eureka/blob/0630b9fb4346fa2d411b7c44ad5299c1d4a4a6c9/part-2/utils/data.py#L99) `to_categorical`: Function to convert categorical data
- [🔗](https://github.com/pouyan9675/ubiquitous-octo-eureka/blob/0630b9fb4346fa2d411b7c44ad5299c1d4a4a6c9/part-2/utils/data.py#L123) `generate_negative_samples`: Function for contrastive learning

### Model Architecture (`utils/model.py`)
- [🔗](https://github.com/pouyan9675/ubiquitous-octo-eureka/blob/0630b9fb4346fa2d411b7c44ad5299c1d4a4a6c9/part-2/utils/model.py#L10) `TextEncoder`: Encodes cell type labels using pretrained biomedical language models
- [🔗](https://github.com/pouyan9675/ubiquitous-octo-eureka/blob/0630b9fb4346fa2d411b7c44ad5299c1d4a4a6c9/part-2/utils/model.py#L38) `PerceiverResampler`: Module for resampling input sequences to a fixed number of latent vectors
- [🔗](https://github.com/pouyan9675/ubiquitous-octo-eureka/blob/0630b9fb4346fa2d411b7c44ad5299c1d4a4a6c9/part-2/utils/model.py#L94) `CrossModalAttention`: Implements cross-attention between modalities
- [🔗](https://github.com/pouyan9675/ubiquitous-octo-eureka/blob/0630b9fb4346fa2d411b7c44ad5299c1d4a4a6c9/part-2/utils/model.py#L201) `FusionModel`: Combines text and gene embeddings in a shared latent space
- [🔗](https://github.com/pouyan9675/ubiquitous-octo-eureka/blob/0630b9fb4346fa2d411b7c44ad5299c1d4a4a6c9/part-2/utils/model.py#L235) `UnifiedMultiModalClassifier`: Main classification model with adversarial component
- [🔗](https://github.com/pouyan9675/ubiquitous-octo-eureka/blob/0630b9fb4346fa2d411b7c44ad5299c1d4a4a6c9/part-2/utils/model.py#L323) `VanillaClassifier`: Baseline model for comparison

### Trainer Infrastructure (`utils/trainer.py`)
- [🔗](https://github.com/pouyan9675/ubiquitous-octo-eureka/blob/0630b9fb4346fa2d411b7c44ad5299c1d4a4a6c9/part-2/utils/trainer.py#L17) `TrainerBase`: Base training framework with logging and device management
- [🔗](https://github.com/pouyan9675/ubiquitous-octo-eureka/blob/0630b9fb4346fa2d411b7c44ad5299c1d4a4a6c9/part-2/utils/trainer.py#L242) `PretrainTrainer`: Handles contrastive pretraining 
- [🔗](https://github.com/pouyan9675/ubiquitous-octo-eureka/blob/0630b9fb4346fa2d411b7c44ad5299c1d4a4a6c9/part-2/utils/trainer.py#L303) `ClassificationTrainer`: Handles adversarial classification training
- [🔗](https://github.com/pouyan9675/ubiquitous-octo-eureka/blob/0630b9fb4346fa2d411b7c44ad5299c1d4a4a6c9/part-2/utils/trainer.py#L561) `VanillaTrainer`: Handles baseline model training

### Loss Functions (`utils/loss.py`)
- [🔗](https://github.com/pouyan9675/ubiquitous-octo-eureka/blob/0630b9fb4346fa2d411b7c44ad5299c1d4a4a6c9/part-2/utils/loss.py#L6) `ContrastiveLoss`: Implementation of contrastive loss for pretraining

### Project Artifacts

- `logs/`: Contains visualizations and training logs
  - `pretraining/`: Visualizations from the pretraining phase
  - `classification/`: Training metrics and model performance visualizations
  - `pretrained_umap.png`: UMAP visualization of the pretrained embeddings
- `weights/`: Contains saved model weights
  - `pretrained.pth`: Pretrained model for after modal alignment
  - `classifier.pth`: Trained classifier for `donor_id` prediction

## Implementation Details

### Data Preparation
- The project loads single-cell data from the CellxGene Census platform
- Filters for human RNA measurements from the central nervous system
- Extracts cell types, donor IDs, sex information, and Geneformer embeddings
- Converts categorical variables and splits data into training and testing sets

### Multimodal Integration
- Text embeddings are generated using the [BioBERT](https://huggingface.co/dmis-lab/biobert-v1.1) model (`dmis-lab/biobert-v1.1`)
- Gene expression data is represented by Geneformer embeddings
- Cross-attention mechanism enables information flow between modalities
- Perceiver Resampler maps variable-length inputs to fixed-size latent representations

### Training Strategy
- **Pretraining phase**: Uses contrastive learning to align text and gene embeddings
- **Classification phase**: Fine-tunes the model to predict donor IDs
- **Adversarial component**: Mitigates the influence of sex as a confounding variable
- **Hyperparameter optimization**: Uses Bayesian optimization for parameter tuning

### Hyperparameter Optimization
- The project uses Optuna for Bayesian optimization of key hyperparameters
- Parameters optimized include:
  - Learning rate
  - Dropout probability
  - Number of latent vectors in the Perceiver Resampler
  - Hidden dimensions in the Perceiver Resampler
  - Weight for adversarial loss (`lambda_adv`)
- Results are captured and the best parameter set is selected for final model training

### Training Parameters
I found the best hyperparameters identified through optimization:

| Parameter | Value |
|-----------|-------|
| `attention_heads` | 2 |
| `num_latent_vectors` | 16 |
| `hidden_size` | 384 |
| `dropout` | 0.3 |
| `learning_rate` | 1e-05 |
| `lambda_adv` | 0.1 |


## Results and Analysis

### Dataset Statistics

The dataset contains 31,780 cells with 60,664 genes. The breakdown of cell types is as follows:

| Cell Type | Count |
|-----------|-------|
| oligodendrocyte | 10,924 |
| cerebellar granule cell | 8,678 |
| microglial cell | 2,562 |
| oligodendrocyte precursor cell | 2,036 |
| GABAergic neuron | 1,744 |
| astrocyte | 1,557 |
| mural cell | 1,076 |
| capillary endothelial cell | 1,072 |
| glutamatergic neuron | 996 |
| endothelial cell of artery | 336 |
| differentiation-committed oligodendrocyte precursor | 306 |
| vascular associated smooth muscle cell | 158 |
| leukocyte | 146 |
| central nervous system macrophage | 110 |
| neuron | 52 |
| ependymal cell | 27 |

### Model Performance

The classification performance metrics for donor ID prediction are as follows:

| Model | Alignment | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|----------|----------|-------------|
| Multimodal w/ Adversarial | ✅ | 58.30% | 0.52 | 0.57 |
| Multimodal w/o Adversarial | ✅ | 58.81% | 0.54 | 0.58 |
| Multimodal w/ Adversarial | ❌ | 61.04% | 0.56 | 0.60 |
| Vanilla Classifier (baseline) | ❌ | **63.04%** | **0.59** | **0.62** |

### Key Findings

1. **Donor Prediction**: All models demonstrated a reasonable ability to predict donor IDs with accuracy significantly above random chance, indicating that single-cell gene expression data contains donor-specific signals.

2. **Pretraining Effect**: Contrary to expectations, the vanilla classifier without pretraining achieved higher accuracy. This could be due to several factors:
   - The pretraining task may not align perfectly with the downstream classification task
   - The more complex model might require more training data or epochs
   - The vanilla model has fewer parameters and might converge faster

3. **Modality Alignment**: While the multimodal approach demonstrated successful embedding alignment in shared space (as shown in the [UMAP visualization](#umap-visualization)), the simpler concatenation approach of the vanilla classifier performed better for classification.

4. **Cross-Attention Analysis**: The visualization of cross-attention weights reveals interesting patterns in how the model attends to different aspects of the modalities.

## Visual Results

### UMAP Visualization
![UMAP Visualization of Cell Types](logs/geneformer_umap.png)
*UMAP projection of cells colored by cell type, based on Geneformer embeddings*

### Embedding Space Visualization
![Embedding Space Visualization](logs/pretrained_umap.png)<a name="umap-visualization"></a>
*Visualization of the shared embedding space after pretraining*

### Training Metrics
![Training Metrics](logs/pretraining/training_metrics.png)
*Pretraining metrics vs. steps*

------

![Training Metrics](logs/classification/training_metrics.png)
*Multimodal Classifier Adversarial: ✅  Alignment: ✅*

![Confusion Matrix](logs/classification/confusion_matrix.png)
*Confusion Matrix of primary trained classifier with modal alignment and adversarial training*

------

![Training Metrics](logs/classification/training_metrics_no_align.png)
*Multimodal Classifier Adversarial: ✅  Alignment: ❌*

------

![Training Metrics](logs/classification/no_adv_training_metrics.png)
*Multimodal Classifier Adversarial: ❌ Alignment: ✅*

------

![Training Metrics](logs/classification/vanilla_training_results.png)
*Vanilla classifier training metrics vs. steps*

### Cross-Attention Visualization

The following are cross attention visualization of the pretrained modal after modal alignment.

![Cross-Attention Visualization](logs/pretraining/cross_attention_visualization.png)
![Cross-Attention Visualization](logs/pretraining/avg_text_to_gene_attention.png)
*Modal cross-attention visualization*

## Conclusion

The project successfully demonstrates that single-cell gene expression data, combined with cell type information, can predict donor identifiers with reasonable accuracy. In addition, the results suggest that **simpler models might  outperform more complex architectures for specific tasks** and they can converge faster as they have less number of parameters and requires less computational overhead.


## Installation and Usage

```bash
# Clone the repository
git clone git@github.com:pouyan9675/ubiquitous-octo-eureka.git
cd part-2

# Install dependencies
pip install -r requirements.txt

# Run jupyter notebook
jupyter notebook
```

Open the notebook `part2.ipynb` to explore the analysis and results.

## Dependencies

- PyTorch
- cellxgene_census
- scanpy
- umap
- transformers
- matplotlib
- seaborn
- optuna
- pandas
- numpy

## Acknowledgments

This project utilizes data from the [CellxGene Census](https://chanzuckerberg.github.io/cellxgene-census/) platform and the [Geneformer](https://geneformer.readthedocs.io/en/latest/) foundation model for single-cell embeddings.