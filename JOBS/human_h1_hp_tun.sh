#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH -D .
#SBATCH -J tfpr_human_h1
#SBATCH -o LOG/human_h1_hp_tun_%A.out
#SBATCH -e LOG/human_h1_hp_tun_%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yiming.kang@wustl.edu

ID=$( echo $(cat /scratch/mblab/woojung/Human_TF_Resp/TFPertRespExplainer/RESOURCES/TI_TFPert/TGI_RNASEQ_TFS.txt))

python3 -u CODE/explain_human_resps.py \
	--model_tuning \
	-i $ID \
	-f tf_binding histone_modifications chromatin_accessibility dna_sequence_nt_freq gene_expression \
	-x /scratch/mblab/woojung/Human_TF_Resp/TFPertRespExplainer/RESOURCES/h5_data/H1_ENCODE_RNASEQ_PC2_k562_genes_GRCh38.h5 \
	-y /scratch/mblab/woojung/Human_TF_Resp/TFPertRespExplainer/RESOURCES/TI_TFPert/TGI_GRCh38_pertResp_DESeq_long.csv \
	-o OUTPUT/human_23tfs_h1_hp_tun

