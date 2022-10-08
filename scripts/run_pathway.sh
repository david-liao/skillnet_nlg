#!/usr/bin/env bash
export PYTHONIOENCODING=utf8

WORKING_DIR=.

CMD="pip install -r ${WORKING_DIR}/requirements.txt"
echo ${CMD}
${CMD}

DATASETS=$1
MODEL=$2
STEPS=$3
BATCH_SIZE=$4
EVAL_BATCH_SIZE=$5
GRAD_ACC=$6
LR=$7
MAX_SRC_LEN=$8
MAX_TGT_LEN=$9
TEMPERATURE=${10}
PLATFORM=${11}
CHECKPOINT=${12}

if [[ $PLATFORM == a100 ]]; then
    PLATFORM="share_733425"
else
    PLATFORM="share_916081"
fi

DATA_NAME=`echo $DATASETS | sed 's/ /./g'`

BART_500K_PATH="/apdcephfs/${PLATFORM}/duyu_shared_data/pretrained_models/zhbart/bart-large/step500000_bs8_lr1e-4_G16_resume_2/checkpoint/500000"
T5_500K_PATH="/apdcephfs/${PLATFORM}/duyu_shared_data/pretrained_models/t5-chinese/t5-base.wwm/500000"

array=(${MODEL//./ })
MODEL_TYPE=${array[0]}
GPU_INFO=${array[1]}
START_POINT=${array[2]}  # pretrain/resume/ckpt/skill
STEP_INFO=${array[3]}

if [[ $MODEL_TYPE == "t5" ]]; then
    CKPT_PATH=$T5_500K_PATH
else
    if [[ $START_POINT == "ckpt" ]]; then
        CKPT_PATH=$CHECKPOINT
    else
        CKPT_PATH=$BART_500K_PATH
    fi
    
    if [[ $START_POINT == "skill" ]]; then
        EXPERT_IDS=$CHECKPOINT
    fi
fi

SAVE_PATH="${WORKING_DIR}/checkpoints/${DATA_NAME}/${MODEL}/step${STEPS}_bs${BATCH_SIZE}_lr${LR}_G${GRAD_ACC}_T${TEMPERATURE}"

OPTS=""
OPTS+=" --model_name_or_path ${CKPT_PATH}"
OPTS+=" --data_path ${WORKING_DIR}/dataset/"
# OPTS+=" --data_name ${DATA_NAME}"
OPTS+=" --datasets ${DATASETS}"
OPTS+=" --output_dir ${SAVE_PATH}"
OPTS+=" --max_source_length ${MAX_SRC_LEN}"
OPTS+=" --max_target_length ${MAX_TGT_LEN}"
OPTS+=" --val_max_target_length ${MAX_TGT_LEN}"
OPTS+=" --do_train"
# OPTS+=" --max_train_samples 128"
# OPTS+=" --do_eval"
OPTS+=" --max_eval_samples 3"
# OPTS+=" --do_predict"
# OPTS+=" --max_predict_samples 128"
# OPTS+=" --num_train_epochs ${EPOCHS}"
OPTS+=" --max_steps ${STEPS}"
OPTS+=" --per_device_train_batch_size ${BATCH_SIZE}"
OPTS+=" --per_device_eval_batch_size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient_accumulation_steps ${GRAD_ACC}"
OPTS+=" --learning_rate ${LR}"
OPTS+=" --logging_steps 100"
# OPTS+=" --evaluation_strategy steps"
# OPTS+=" --eval_steps 5000"
OPTS+=" --save_strategy steps"
OPTS+=" --save_steps 5000"
# OPTS+=" --warmup_steps 10000"
OPTS+=" --warmup_ratio 0.1"
OPTS+=" --disable_tqdm True"
# OPTS+=" --load_best_model_at_end True"
# OPTS+=" --metric_for_best_model bleu"

# OPTS+=" --save_total_limit 3"
OPTS+=" --logging_dir ${SAVE_PATH}/runs"
OPTS+=" --fp16"
OPTS+=" --num_beams 4"
OPTS+=" --predict_with_generate"
OPTS+=" --temperature ${TEMPERATURE}"
OPTS+=" --model_type ${MODEL_TYPE}"

if [[ $MODEL_TYPE == "pathway" ]]; then
    OPTS+=" --topk_experts -1"
fi

if [[ $START_POINT == "resume" ]]; then
    OPTS+=" --resume_from_checkpoint ${WORKING_DIR}/${CHECKPOINT}"
fi

if [[ $START_POINT == "skill" ]]; then
    OPTS+=" --expert_ids ${EXPERT_IDS}"
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

~/cache.sh