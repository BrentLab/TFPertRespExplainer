#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH -o LOG/score_bp_%A_%a.out
#SBATCH -e LOG/score_bp_%A_%a.err
#SBATCH -J score_bp

## Example:
# sbatch --array=1-196 HELPER_SCRIPTS/score_binding_potentials.sh RESOURCES/Yeast_PWMs/ScerTF_PWM_list.txt RESOURCES/Yeast_PWMs/ScerTF RESOURCES/Yeast_genome/S288C_R64_sacCer3_deGoer2020_tss1000to500b.fa OUTPUT/Yeast_bp_tss1000to500b

# Input variables
TF_LIST_FILENAME=$1     # Filename of TF list
PWM_DIRNAME=$2          # Directory of TF PWM
REG_DNA_FILENAME=$3     # Filename of regulatory DNA
FIMO_OUT_DIRNAME=$4     # Directory of FIMO output 

read tf < <( sed -n ${SLURM_ARRAY_TASK_ID}p $TF_LIST_FILENAME )
set -e

if [[ ! -z ${tf} ]]; then
    if [ -f $PWM_DIRNAME/$tf ]; then
        fimo -o $FIMO_OUT_DIRNAME/$tf --thresh 5e-3 $PWM_DIRNAME/$tf $REG_DNA_FILENAME
    else
        printf "No PWM available for %s\n" $tf
    fi
fi
