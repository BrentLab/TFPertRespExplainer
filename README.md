# TF-Perturbation Response Explainer

The ability to predict which genes will respond to perturbation of a transcription factor (TF)'s activity serves as a benchmark for our systems-level understanding of transcriptional regulatory networks. This repo uses machine learning models to predict each gene's responsiveness to a TF perturbation using genomic data from the unperturbed cells, and explains which genomic factors determine the model predictions.

## Installation

Use [miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage packages.

```
conda create -n tfpr_exp python=3.6.10
conda activate tfpr_exp
conda config --append channels conda-forge --append channels bioconda
conda install --file requirements.txt
```

## Usage

### Explaining a gene's responsiveness to a TF perturbation

`TFPRExplainer` predicts which gene would response to a TF perturbation and explains the extend to which each feature contributes to the prediction of responsiveness. Use the following example code to load feature matrix, cross validate response predictions across TFs, and analyze contributions of relevant genomic features.

It takes two forms of parameters:

- *Command line arguments* define the perturbed TF, the collection of genomic features, and directory paths for input and output data. Use `-h` for details.
- *Configuration parameters* define the boundary of each gene's regulatory DNA, and instructions on processing feature matrix and response label. Reference `config.ini` for default parameters.

For yeast genome, run

```
$ python3 CODE/explain_yeast_resps.py \
    -i <tf1 tf2 ... tfn> \
    -f tf_binding histone_modifications chromatin_accessibility dna_sequence_nt_freq gene_expression gene_variation \
    -x <feature matrix (.h5)> \
    -y <tf-pert response data (.csv)> \
    -o <output directory>
```

For human genome, use `CODE/explain_human_resps.py` with the above set of arguments.

### Visualizing feature contributions in Jupyter notebooks

Use Jupyter notebooks in `Notebooks/` to make feautre visualization.


## Input Data

### Features

The feature matrix data are store in a hdf5 file. Use the pre-compiled hdf5 file or build your own using the following code for yeast data. Use `-h` argument for argument details.

```
$ python3 CODE/preprocess_yeast_data.py \
    -o <feature matrix (.h5)> \
    -a <tss data (.bed)> \
    -f <reference genome (.fa)> \
    --tf_bind <tf binding peaks (.bed)> \
    --hist_mod <histone marks data (.bed)> \
    --chrom_acc <chromatin accessibility data (.bed)> \
    --gene_expr <pre-pert gex levels (.csv)> \
    --gene_var <pre-pert gex variations (.csv)> 
```

For human data, use `CODE/preprocess_human_data.py` with the above set of arguments with the additional `-r` for distal enhancer and promoter data.

### Response label

Store the magnitude of genes' responses to perturbations in wide or long format.

## Output Data

- `stats`: Overall performance of cross-validation.
- `preds`: Predicted probability of being responsive for each gene.
- `feat_shap_wbg`: A matrix of feature contributions (SHAP values) in dimension of gene x feature. Each entry explains the extend to which a feature contributes to predict a gene's responsiveness.
- `feats`: Feature names and their corresponding ranges of column indices in `feat_shap_wbg`.
- `genes`: Gene names corresponding to row indices in `feat_shap_wbg`.
- `feat_mtx`: Feature matrix (gene x feature) constructed from input hdf5.

## References
- Kang Y, Jung WJ, Brent MR. 2022. Predicting which genes will respond to transcription factor perturbations. G3 Genes| Genomes| Genetics. [doi:10.1093/g3journal/jkac144](https://doi.org/10.1093/g3journal/jkac144)
