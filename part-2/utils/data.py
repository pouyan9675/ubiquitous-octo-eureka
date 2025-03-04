import random
import logging
from typing import Dict, Any

import pandas as pd
import torch
import cellxgene_census
from torch.utils.data import Dataset


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CellDataset(Dataset):
    """
    A custom PyTorch Dataset for handling cell data from an AnnData object.

    This dataset stores cell types, sex, donor IDs, and geneformer embeddings, 
    converting them into PyTorch tensors where applicable for use in machine learning models.

    Parameters:
    -----------
    data : dict
        A dictionary containing:
        - "cell_type": List of cell type labels.
        - "sex": List or array of categorical values representing sex, converted to `torch.long`.
        - "geneformer_embeddings": List or array of embeddings, converted to `torch.float32`.
        - "donor_id": List or array of donor IDs, converted to `torch.long`.

    Returns:
    --------
    dict:
        A dictionary containing:
        - "cell_type": Cell type label at index `idx`.
        - "sex": Encoded sex tensor at index `idx`.
        - "geneformer_embeddings": Embedding tensor at index `idx`.
        - "donor_id": Donor ID tensor at index `idx`.
    """
    
    def __init__(self, data: Dict[str, Any]):
        self.cell_type = data["cell_type"]
        self.sex = torch.tensor(data["sex"], dtype=torch.long)
        self.geneformer_embeddings = torch.tensor(data["geneformer_embeddings"], dtype=torch.float32)
        self.donor_id = torch.tensor(data["donor_id"], dtype=torch.long)

    def __len__(self):
        return len(self.donor_id)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "cell_type": self.cell_type[idx],
            "sex": self.sex[idx],
            "geneformer_embeddings": self.geneformer_embeddings[idx],
            "donor_id": self.donor_id[idx]
        }


def load_ann_data(census_version="2023-12-15", 
                emb_names=["geneformer"],
                obs_column_names=["cell_type", "donor_id", "sex"],
                organism="homo_sapiens",
    ):
    """
    Load an AnnData object from the CellxGene Census dataset based on specified filters.

    Parameters:
    -----------
    census_version : str, optional
        The version of the CellxGene Census dataset to use. Default is "2023-12-15" for reproducibility.
    emb_names : list of str, optional
        A list of embedding names to include in the observation matrix. Default is ["geneformer"].

    Returns:
    --------
    anndata.AnnData
        The loaded AnnData object containing cells from the central nervous system.
    
    """
    with cellxgene_census.open_soma(census_version=census_version) as census:
        # Load AnnData with specified filters
        adata = cellxgene_census.get_anndata(
            census,
            organism=organism,
            measurement_name="RNA",
            obs_value_filter="tissue_general == 'central nervous system'",
            obs_column_names=obs_column_names,
            obs_embeddings=emb_names,
        )

    # Log dataset statistics
    logger.info(f"Loaded {adata.n_obs} cells with {adata.n_vars} genes.")
    logger.info(f"Metadata columns: {list(adata.obs.columns)}")
    logger.info(f"Available embeddings: {list(adata.obsm.keys())}")

    return adata


def to_categorical(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Convert a column in a DataFrame to categorical values.

    Parameters:
    -----------
    data : pandas.DataFrame
        The input DataFrame.
    column_name : str
        The name of the column to convert to categorical values.

    Returns:
    --------
    pandas.DataFrame
        The DataFrame with the specified column converted to categorical values.
    """
    cat = pd.Categorical(df[column_name])
    df[column_name] = cat.codes  # Convert to categorical codes

    # Create mapping from encoded values to original labels
    mapping = {i: cat.categories[i] for i in range(len(cat.categories))}
    return df, mapping


def generate_negative_samples(
    batch: list, 
    target_class: str, 
    labels: torch.Tensor,
    negative_sample_factor: int = 1,
    ) -> torch.Tensor:
    """
    Generate negative samples for contrastive learning.

    Parameters:
    -----------
    batch : list
        The input batch of samples.
    target_class : str
        The target class to change from labels.
    labels : torch.Tensor
        The labels for the input batch. In this case is cell type.

    Returns:
    --------
    torch.Tensor
        The batch of negative samples.
    """
    negative_samples = []
    
    for f in range(negative_sample_factor):
        for i in range(len(batch)):
            label = batch[i][target_class]  # get target class
            l = random.choice(labels)   # select random label
            while l == label:           # ensure different labels
                l = random.choice(labels)
            n_sample = batch[i].copy()
            n_sample[target_class] = l
            negative_samples.append(n_sample)
            
    return negative_samples