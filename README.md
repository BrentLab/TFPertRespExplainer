# Perturbation Response Modeling

The ability to predict which genes will respond to perturbation of a TF's activity serves as a benchmark for our systems-level understanding of transcriptional regulatory networks. This repo uses machine learning models to predict each gene's responsiveness to a TF perturbation using genomic data from the unperturbed cells, and explains which genomic factors determine the model predictions.

## Installation

Use `miniconda` to manage packages. If on SLURM cluster, use `module load miniconda3`; otherwise, install and configure your own conda manager. 

```
conda create -n tfpr_exp python=3.6.10
conda activate tfpr_exp
conda config --append channels conda-forge 
conda config --append channels bioconda
conda install numpy==1.19.2 pandas==1.1.3 scikit-learn==0.22.1 jupyterlab==2.2.6 jedi==0.17.2 pybedtools==0.8.0 biopython==1.78 h5py==2.10.0 multiprocess==0.70.11.1 xgboost==0.90 shap==0.35.0 plotnine==0.7.1
```

## Usage

### Explaining a gene's responsiveness to a TF perturbation

`TFPRExplainer` predicts which gene would response to a TF perturbation and explains the extend to which each feature contributes to predicting the responsiveness. Use the following code to load feature matrix from hdf5 file, cross validate response predictions, and analyze contributions of relevant genomic features

```
$ python3 CODE/explain_yeast_resps.py \
    -i YLR451W \
    -x OUTPUT/h5_data/yeast_dna_cc_hm_atac_tss1000to500b_expr_var.h5 \
    -y RESOURCES/Yeast_ZEV_IDEA/ZEV_15min_shrunkenData.csv \
    -f tf_binding histone_modifications chromatin_accessibility dna_sequence_nt_freq gene_expression gene_variation \
    -o OUTPUT/Yeast_CallingCards_ZEV/all_feats/xgb
```

### Explaining a gene's frequency of response across perturbations

```
$ python3 CODE/explain_yeast_resps.py \
    --is_regressor \
    -i freq \
    -x OUTPUT/h5_data/yeast_dna_cc_hm_atac_tss1000to500b_expr_var.h5 \
    -y RESOURCES/Yeast_ZEV_IDEA/ZEV_15min_shrunkenData_deFreq.csv \
    -f histone_modifications chromatin_accessibility dna_sequence_nt_freq gene_expression gene_variation \
    -o OUTPUT/Yeast_ZEV_DE_freq/tf_indep_feats/xgb/
```

### Visualizing feature contributions in Jupyter notebooks

Use Jupyter notebooks in `Notebooks/` to make corresponding feautre visualization.


## Input Data

### Features

Use pre-compiled hdf5 files or compile on your own using the following code as an example.

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

## Output Data
