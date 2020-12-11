# Perturbation Response Modeling

## Installation

Use `miniconda` to manage packages. On HTCF cluster, load it as a module using `ml miniconda3`; or install and configure your own conda manager. 

```
conda create -n pert_resp python=3.6.10
conda activate pert_resp
conda config --channel conda-forge --channel bioconda
conda install numpy pandas scikit-learn ipython pybedtools biopython h5py multiprocess xgboost
conda deactivate
```

## Usage

### 1. Define modeling parameters.

```
UP_BOUND=1000
DOWN_BOUND=500
BINS=15
ALGORITHM=xgb_classifier

LABEL_FILE=RESOURCES/ZEV_15min_shrunkenData.csv
FEAT_FILE=OUTPUT/YEAST/FeatData.h5
OUTPUT_DIR=OUTPUT/YEAST/tss${UP_BOUND}to${DOWN_BOUND}b_${BINS}bins
```

### 2. Prepare resource data into hdf5 file.

```
python3 CODE/preprocess_data.py -o OUTPUT/h5_data/yeast_dna_cc_hm_atac_${UP_BOUND}to${DOWN_BOUND}b_expr.h5 -a RESOURCES/Yeast_genome/S288C_R64-1-1_sacCer3_orf.bed -f RESOURCES/Yeast_genome/S288C_R64-1-1_sacCer3_genome.fa --feat_bound ${UP_BOUND} ${DOWN_BOUND} --tf_bind RESOURCES/Yeast_CallingCards/*.bed --hist_mod RESOURCES/Yeast_HistoneMarks_Weiner2014/*.bed --chrom_acc RESOURCES/Yeast_ChromAcc_Schep2015/BY4741_ypd_osm_0min.occ.bed --gene_expr RESOURCES/Yeast_ZEV_IDEA/ZEV_0min_prepertData.csv 
```

### 3. Train and test the model for each TF using corss-validation.

```
python3 -u CODE/cv_yeast_model.py -m cv_model -i $TF -a $ALGORITHM \
	-x $FEAT_FILE -y $LABEL_FILE --feat_bins ${BINS} --feat_bound ${UP_BOUND} ${DOWN_BOUND} \
	-f tf_binding histone_modifications chromatin_accessibility gene_expression gene_variation dna_sequence_nt_freq \
	-o $OUTPUT_DIR/
```

### 4. Explain the determinants of TF-perturbation responsiveness
```
python3 -u CODE/interpret_model.py -a $ALGORITHM \
	-m ${TF_MODEL_DIR}/cv*.pkl \
	-x ${TF_MODEL_DIR}/feat_mtx.csv.gz \
	-y ${TF_MODEL_DIR}/preds.csv.gz \
	-g ${TF_MODEL_DIR}/genes.csv.gz \
	-o ${TF_MODEL_DIR}
```
