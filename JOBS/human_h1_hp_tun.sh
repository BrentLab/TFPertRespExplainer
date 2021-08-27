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

#python3 -u CODE/explain_human_resps.py \
#	--model_tuning \
#	-i $ID \
#	-f tf_binding histone_modifications chromatin_accessibility dna_sequence_nt_freq gene_expression \
#	-x /scratch/mblab/woojung/Human_TF_Resp/TFPertRespExplainer/RESOURCES/h5_data/H1_ENCODE_RNASEQ_PC2_k562_genes_GRCh38.h5 \
#	-y /scratch/mblab/woojung/Human_TF_Resp/TFPertRespExplainer/RESOURCES/TI_TFPert/TGI_GRCh38_pertResp_DESeq_long.csv \
#	-o OUTPUT/human_23tfs_h1_hp_tun

python3 -u CODE/explain_human_resps.py \
	-i ENSG00000100811 ENSG00000105698 ENSG00000106459 ENSG00000111704 ENSG00000115966 ENSG00000118260 ENSG00000119950 ENSG00000120738 ENSG00000136997 ENSG00000140262 ENSG00000143390 ENSG00000158773 ENSG00000169016 ENSG00000171606 ENSG00000172216 ENSG00000175592 ENSG00000177045 ENSG00000177606 ENSG00000197905 ENSG00000198517 \
	-f tf_binding histone_modifications chromatin_accessibility dna_sequence_nt_freq gene_expression \
	-x /scratch/mblab/woojung/Human_TF_Resp/TFPertRespExplainer/RESOURCES/h5_data/H1_ENCODE_RNASEQ_PC2_k562_genes_GRCh38.h5 \
	-y /scratch/mblab/woojung/Human_TF_Resp/TFPertRespExplainer/RESOURCES/TI_TFPert/TGI_GRCh38_pertResp_DESeq_long.csv \
	-o OUTPUT/human_23tfs_h1_hp_base

