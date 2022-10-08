#!/usr/bin/env bash
export PYTHONIOENCODING=utf8

WORKING_DIR=.

# CMD="pip install -r ${WORKING_DIR}/requirements.txt"
# echo ${CMD}
# ${CMD}

PHASE=$1
DATASETS=$2
MODEL=$3
# STEPS=$3
# BATCH_SIZE=$4
EVAL_BATCH_SIZE=$4
# GRAD_ACC=$6
# LR=$7
MAX_SRC_LEN=$5
MAX_TGT_LEN=$6
# TEMPERATURE=${10}
# PLATFORM=${11}
NUM_BEAMS=$7
METRIC=$8

# if [[ $PLATFORM == a100 ]]; then
#     PLATFORM="share_733425"
# else
#     PLATFORM="share_916081"
# fi

if [[ $MODEL =~ "bart" ]]; then
    MODEL_TYPE="bart"
elif [[ $MODEL =~ "moe" ]]; then
    MODEL_TYPE="moe"
elif [[ $MODEL =~ "pathway" ]]; then
    MODEL_TYPE="pathway"
    TOPK_EXPERTS=-1
else
    MODEL_TYPE="t5"
fi

DATA_NAME=`echo $DATASETS | sed 's/ /./g'`
# CKPT_PATH=checkpoints/adgen.kdconv.lcsts.matinf.nlpcc/bart_500k_gpu32/$MODEL
CKPT_PATH=$MODEL
SAVE_PATH=$CKPT_PATH/$PHASE/$DATA_NAME

OPTS=""
OPTS+=" --model_name_or_path ${CKPT_PATH}"
OPTS+=" --data_path ${WORKING_DIR}/dataset"
OPTS+=" --datasets ${DATASETS}"
OPTS+=" --output_dir ${SAVE_PATH}"
OPTS+=" --max_source_length ${MAX_SRC_LEN}"
# OPTS+=" --max_target_length ${MAX_TGT_LEN}"
OPTS+=" --val_max_target_length ${MAX_TGT_LEN}"
# OPTS+=" --do_train"
# OPTS+=" --max_train_samples 128"
# OPTS+=" --do_eval"
# OPTS+=" --max_eval_samples 3"
# OPTS+=" --do_predict"
# OPTS+=" --max_predict_samples 128"
OPTS+=" --do_${PHASE}"
# OPTS+=" --num_train_epochs ${EPOCHS}"
# OPTS+=" --max_steps ${STEPS}"
# OPTS+=" --per_device_train_batch_size ${BATCH_SIZE}"
OPTS+=" --per_device_eval_batch_size ${EVAL_BATCH_SIZE}"
# OPTS+=" --gradient_accumulation_steps ${GRAD_ACC}"
# OPTS+=" --learning_rate ${LR}"
# OPTS+=" --logging_steps 100"
# OPTS+=" --evaluation_strategy steps"
# OPTS+=" --eval_steps 5000"
# OPTS+=" --save_strategy steps"
# OPTS+=" --save_steps 5000"
# OPTS+=" --warmup_steps 10000"
# OPTS+=" --disable_tqdm True"
# OPTS+=" --load_best_model_at_end True"
OPTS+=" --metric_for_best_model ${METRIC}"

# OPTS+=" --save_total_limit 3"
OPTS+=" --logging_dir ${SAVE_PATH}/runs"
OPTS+=" --fp16"
OPTS+=" --num_beams ${NUM_BEAMS}"
OPTS+=" --predict_with_generate"
# OPTS+=" --temperature ${TEMPERATURE}"
OPTS+=" --report_to none"
OPTS+=" --model_type ${MODEL_TYPE}"

if [ -n "${TOPK_EXPERTS}" ]; then
    OPTS+=" --topk_experts ${TOPK_EXPERTS}"
fi


# OPTS+=" --overwrite_cache True"


my_port=23456
# CMD="python3 -u -m light.pytorch.launch --nproc_per_node=8 --use_env ${WORKING_DIR}/code/run_multitask.py ${OPTS}"
# CMD="python3 -u -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node $HOST_GPU_NUM --master_addr $CHIEF_IP --master_port $my_port ${WORKING_DIR}/code/run_multitask.py ${OPTS}"
CMD="python3 -u -m torch.distributed.launch --nnodes=$HOST_NUM --node_rank=$INDEX --nproc_per_node $HOST_GPU_NUM --master_addr $CHIEF_IP --master_port $my_port ${WORKING_DIR}/run/run_multitask.py ${OPTS}"


echo ${CMD}
mkdir -p ${SAVE_PATH}

export NCCL_IB_DISABLE=1
${CMD} 2>&1 | tee ${SAVE_PATH}/train_${INDEX}.log

# ~/cache.sh