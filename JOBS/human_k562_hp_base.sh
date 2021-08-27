#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH -D .
#SBATCH -J tfpr_k562_base
#SBATCH -o LOG/human_k562_hp_base_%A.out
#SBATCH -e LOG/human_k562_hp_base_%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yiming.kang@wustl.edu

python3 -u CODE/explain_human_resps.py \
	--disable_shape \
	-i ENSG00000116017 ENSG00000162772 ENSG00000156273 ENSG00000134107 ENSG00000115816 ENSG00000102974 ENSG00000205250 ENSG00000169016 ENSG00000105722 ENSG00000141568 ENSG00000111206 ENSG00000102145 ENSG00000172273 ENSG00000147421 ENSG00000130522 ENSG00000197063 ENSG00000198517 ENSG00000125952 ENSG00000103495 ENSG00000134138 ENSG00000187098 ENSG00000119950 ENSG00000082641 ENSG00000001167 ENSG00000177463 ENSG00000185551 ENSG00000123358 ENSG00000143390 ENSG00000130254 ENSG00000143379 ENSG00000177045 ENSG00000113658 ENSG00000185591 ENSG00000112658 ENSG00000126561 ENSG00000162367 ENSG00000074219 ENSG00000197905 ENSG00000198176 ENSG00000131931 ENSG00000158773 ENSG00000105698 ENSG00000136451 ENSG00000060138 ENSG00000144161 ENSG00000166478 ENSG00000126746 ENSG00000186918 \
	-f tf_binding histone_modifications chromatin_accessibility dna_sequence_nt_freq gene_expression gene_variation \
	-x /scratch/mblab/yiming.kang/Pert_Response_Modeling/OUTPUT/h5_data/human_encode_enhan_alltss_2kbto2kb_promo.h5 \
	-y /scratch/mblab/yiming.kang/Pert_Response_Modeling/RESOURCES/Human_TFPert_ENCODE/K562_pertResp_DESeq2_long.csv \
	-o OUTPUT/human_k562_hp_base

