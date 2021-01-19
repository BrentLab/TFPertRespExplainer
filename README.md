# Perturbation Response Modeling

## Installation

Use `miniconda` to manage packages. If on SLURM cluster, use `module load miniconda3`; otherwise, install and configure your own conda manager. 

```
conda create -n tfpr_exp python=3.6.10
conda activate tfpr_exp
conda config --append channels conda-forge 
conda config --append channels bioconda
conda install numpy pandas scikit-learn==0.23.2 jupyterlab==2.2.6 jedi==0.17.2 pybedtools biopython h5py multiprocess xgboost==1.3.0 shap==0.37.0 plotnine
conda deactivate
```

## Usage

### 1. Prepare resource data and save as hdf5 file.

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

### 2. Explain genomic features that are predictive of which genes would respond to TF perturbation.

```
$ python3 -u CODE/predict_yeast_resps.py \
    -i YLR451W \
    -x OUTPUT/h5_data/yeast_dna_cc_hm_atac_tss1000to500b_expr_var.h5 \
    -y RESOURCES/Yeast_ZEV_IDEA/ZEV_15min_shrunkenData.csv \
    -f tf_binding histone_modifications chromatin_accessibility dna_sequence_nt_freq gene_expression gene_variation \
    -o OUTPUT/Yeast_CallingCards_ZEV/all_feats/xgb
```

### 3. Visualize feature contributions in Jupyter notebooks.

```
$ jupyter lab
```
