# TF-Perturbation Response Explainer

The ability to predict which genes will respond to perturbation of a TF's activity serves as a benchmark for our systems-level understanding of transcriptional regulatory networks. This repo uses machine learning models to predict each gene's responsiveness to a TF perturbation using genomic data from the unperturbed cells, and explains which genomic factors determine the model predictions.

## Installation

Use `miniconda` to manage packages. If on HTCF cluster (SLURM), use `module load miniconda3`; otherwise, install and configure your own conda manager. 

```
conda create -n tfpr_exp python=3.6.10
conda activate tfpr_exp
conda config --append channels conda-forge 
conda config --append channels bioconda
conda install numpy==1.19.2 pandas==1.1.3 scikit-learn==0.22.1 jupyterlab==2.2.6 jedi==0.17.2 pybedtools==0.8.0 biopython==1.78 h5py==2.10.0 multiprocess==0.70.11.1 xgboost==0.90 shap==0.35.0 plotnine==0.7.1
```

## Usage

### Explaining a gene's responsiveness to a TF perturbation

`TFPRExplainer` predicts which gene would response to a TF perturbation and explains the extend to which each feature contributes to the prediction of responsiveness. Use the following example code to load feature matrix, cross validate response predictions, and analyze contributions of relevant genomic features.

It takes two forms of parameters:

- *Command line arguments* define the perturbed TF, the collection of genomic features, and directory paths for input and output data. Use `--help` for details.
- *Configuration parameters* define the boundary of each gene's regulatory DNA, and instructions on processing feature matrix and response label. Reference `config.ini` for default parameters.

For yeast genome, run

```
$ python3 CODE/explain_yeast_resps.py \
    -i YLR451W \
    -f tf_binding histone_modifications chromatin_accessibility dna_sequence_nt_freq gene_expression gene_variation \
    -x RESOURCES/h5_data/yeast_dna_cc_hm_atac_tss1000to500b_expr_var.h5 \
    -y RESOURCES/Yeast_ZEV_IDEA/ZEV_15min_shrunkenData.csv \
    -o OUTPUT/Yeast_CallingCards_ZEV/all_feats/
```

For human genome, run

```
$ python3 CODE/explain_human_resps.py \
    -i ENSG00000001167 \
    -f tf_binding histone_modifications chromatin_accessibility dna_sequence_nt_freq gene_expression gene_variation \
    -x RESOURCES/h5_data/human_encode_enhan_alltss_2kbto2kb_promo.h5 \
    -y RESOURCES/HumanK562_TFPert/K562_pertResp_DESeq2_long.csv \
    -o OUTPUT/Human_ChIPseq_TFpert//all_feats/
```

### Explaining a gene's frequency of response across perturbations

```
$ python3 CODE/explain_yeast_resps.py \
    --is_regressor \
    -i freq \
    -f histone_modifications chromatin_accessibility dna_sequence_nt_freq gene_expression gene_variation \
    -x RESOURCES/h5_data/yeast_dna_cc_hm_atac_tss1000to500b_expr_var.h5 \
    -y RESOURCES/Yeast_ZEV_IDEA/ZEV_15min_shrunkenData_deFreq.csv \
    -o OUTPUT/Yeast_ZEV_DE_freq/tf_indep_feats/xgb/
```

### Visualizing feature contributions in Jupyter notebooks

Use Jupyter notebooks in `Notebooks/` to make feautre visualization.


## Input Data

### Features

All feature data are store in one hdf5 file. Use the pre-compiled hdf5 file or compile on your own using the following code as an example for yeast data.

```
$ python3 CODE/preprocess_data.py \
    -o OUTPUT/h5_data/yeast_s288c_data.h5 \
    -a RESOURCES/Yeast_genome/S288C_R64-1-1_sacCer3_orf.bed \
    -f RESOURCES/Yeast_genome/S288C_R64-1-1_sacCer3_genome.fa \
    --tf_bind RESOURCES/Yeast_CallingCards/*.bed \
    --hist_mod RESOURCES/Yeast_HistoneMarks_Weiner2014/*.bed \
    --chrom_acc RESOURCES/Yeast_ChromAcc_Schep2015/BY4741_ypd_osm_0min.occ.bed \
    --gene_expr RESOURCES/Yeast_ZEV_IDEA/[...].csv \
    --gene_var RESOURCES/Yeast_ZEV_IDEA/[...].csv 
```

### Response label

Store the magnitude of genes' responses to perturbations in wide or long format.

## Output Data

- `stats`: Overall performance of cross-validation.
- `preds`: Predicted probability of being responsive for each gene.
- `feat_shap_wbg`: A matrix of feature contributions (SHAP values) in dimension of gene x feature. Each entry explains the extend to which a feature contributes to predict a gene's responsiveness.
- `feats`: Feature names and their corresponding ranges of column indices in `feat_shap_wbg`.
- `genes`: Gene names corresponding to row indices in `feat_shap_wbg`.
- `feat_mtx`: Feature matrix (gene x feature) constructed from input hdf5.
