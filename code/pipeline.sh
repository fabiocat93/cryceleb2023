#!/bin/bash                      
#SBATCH --job-name=cryceleb
#SBATCH --output=../logs/%j.out
#SBATCH --error=../logs/%j.err
#SBATCH -t 00:30:00          # walltime = 1 hours and 30 minutes
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fabiocat@mit.edu
# Set the array variable based on the calculated array size
#SBATCH --array=0-350%1

# Execute commands to run your program here. Here is an example of python.
eval "$(conda shell.bash hook)"
conda activate fab

# Define the arrays of encoders and metrics
encoders=("ecapa-voxceleb-ft-cryceleb" "spkrec-ecapa-voxceleb" "human_cochleagram" "log-mel-spectrogram" "pyannote-embedding" "serab_byols" "apc" "tera" "hubert_0" "hubert_1" "hubert_2" "hubert_3" "hubert_4" "hubert_5" "hubert_6" "hubert_7" "hubert_8" "hubert_9" "hubert_10" "hubert_11" "hubert_12" "hubert_13" "hubert_14" "hubert_15" "hubert_16" "hubert_17" "hubert_18" "hubert_19" "hubert_20" "hubert_21" "hubert_22" "hubert_23" "hubert_24" "hubert_25" "hubert_26" "hubert_27" "hubert_28" "hubert_29" "hubert_30" "hubert_31" "hubert_32" "hubert_33" "hubert_34" "hubert_35" "hubert_36" "hubert_37" "hubert_38" "hubert_39" "hubert_40" "hubert_41" "hubert_42" "hubert_43" "hubert_44" "hubert_45" "hubert_46" "hubert_47" "hubert_48" "wav2vec2_0" "wav2vec2_1" "wav2vec2_2" "wav2vec2_3" "wav2vec2_4" "wav2vec2_5" "wav2vec2_6" "wav2vec2_7" "wav2vec2_8" "wav2vec2_9" "wav2vec2_10" "wav2vec2_11" "wav2vec2_12" "wav2vec2_13" "wav2vec2_14" "wav2vec2_15" "wav2vec2_16" "wav2vec2_17" "wav2vec2_18" "wav2vec2_19" "wav2vec2_20" "wav2vec2_21" "wav2vec2_22" "wav2vec2_23" "wav2vec2_24" "data2vec2_0" "data2vec2_1" "data2vec2_2" "data2vec2_3" "data2vec2_4" "data2vec2_5" "data2vec2_6" "data2vec2_7" "data2vec2_8" "data2vec2_9" "data2vec2_10" "data2vec2_11" "data2vec2_12" "data2vec2_13" "data2vec2_14" "data2vec2_15" "data2vec2_16" "data2vec2_17" "data2vec2_18" "data2vec2_19" "data2vec2_20" "data2vec2_21" "data2vec2_22" "data2vec2_23" "data2vec2_24" "bookbot-wav2vec2-adult-child-cls_0" "bookbot-wav2vec2-adult-child-cls_1" "bookbot-wav2vec2-adult-child-cls_2" "bookbot-wav2vec2-adult-child-cls_3" "bookbot-wav2vec2-adult-child-cls_4" "bookbot-wav2vec2-adult-child-cls_5" "bookbot-wav2vec2-adult-child-cls_6" "bookbot-wav2vec2-adult-child-cls_7" "bookbot-wav2vec2-adult-child-cls_8" "bookbot-wav2vec2-adult-child-cls_9" "bookbot-wav2vec2-adult-child-cls_10" "bookbot-wav2vec2-adult-child-cls_11" "bookbot-wav2vec2-adult-child-cls_12")
metrics=("cosine" "euclidean" "manhattan")

# Get the index for the current task in the array
encoder_index=$((SLURM_ARRAY_TASK_ID / ${#metrics[@]}))
metric_index=$((SLURM_ARRAY_TASK_ID % ${#metrics[@]}))

# Retrieve the encoder and metric for the current task
ENCODER=${encoders[$encoder_index]}
METRIC=${metrics[$metric_index]}

# Print the current task information
echo "Running task $SLURM_ARRAY_TASK_ID with encoder: $ENCODER and metric: $METRIC"


python my_notebook.py --encoder "$ENCODER" --device "gpu" --metric "$METRIC"