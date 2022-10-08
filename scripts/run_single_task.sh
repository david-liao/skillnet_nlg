#!/usr/bin/env bash
export PYTHONIOENCODING=utf8

WORKING_DIR=.

# CMD="pip install -r ${WORKING_DIR}/requirements.txt"
# echo ${CMD}
# ${CMD}

DATASETS=$1
MODEL=$2
EPOCHS=$3
BATCH_SIZE=$4
EVAL_BATCH_SIZE=$5
GRAD_ACC=$6
LR=$7
MAX_SRC_LEN=$8
MAX_TGT_LEN=$9
METRIC=${10}
CHECKPOINT=${11}
CUDA=${12}


array=(${MODEL//./ })
MODEL_TYPE=${array[0]}
GPU_INFO=${array[1]}
SETTING=${array[2]}     # ckpt/hs(hyperparameter search)/skill
CKPT_NUM=${array[3]}

if [[ $MODEL_TYPE == "bart" ]]; then
    CKPT_PATH=/apdcephfs/share_916081/duyu_shared_data/pretrained_models/zhbart/bart-large/step500000_bs8_lr1e-4_G16_resume_2/checkpoint/500000
elif [[ $MODEL_TYPE == "dense" ]]; then
    MODEL_TYPE="bart"
    CKPT_PATH=/apdcephfs/share_916081/jwliao/experiment/pathway/checkpoints/adgen.kdconv.lcsts.matinf.nlpcc/bart_500k_gpu32/step100000_bs16_lr3e-5_G1_T4/checkpoint-100000
elif [[ $MODEL_TYPE == "moe" ]]; then
    CKPT_PATH=/apdcephfs/share_916081/jwliao/experiment/pathway/checkpoints/adgen.kdconv.lcsts.matinf.nlpcc/moe_gpu32/step100000_bs8_lr3e-5_G2_T4/checkpoint-85000
elif [[ $MODEL_TYPE == "pathway" ]]; then
    if [ ! -n "$SETTING" ] || [ "$SETTING" = "hs" ]; then
        CKPT_PATH=/apdcephfs/share_916081/duyu_shared_data/jwliao/pathway/checkpoints/adgen.kdconv.lcsts.matinf.nlpcc/pathway.gpu64.pretrain.500k/step100000_bs8_lr3e-5_G1_T4/checkpoint-100000
    elif [[ $SETTING =~ "ckpt" ]]; then
        CKPT_PATH=$CHECKPOINT
        MODEL="${MODEL_TYPE}.${GPU_INFO}.${SETTING}/${CKPT_NUM}"
    elif [[ $SETTING == "skill" ]]; then
        CKPT_PATH=/apdcephfs/share_916081/duyu_shared_data/jwliao/pathway/checkpoints/adgen.kdconv.lcsts.matinf.nlpcc/pathway.gpu64.pretrain.500k/step100000_bs8_lr3e-5_G1_T4/checkpoint-100000
        EXPERT_IDS=$CHECKPOINT
    fi
else
    CKPT_PATH=/apdcephfs/share_916081/duyu_shared_data/pretrained_models/t5-chinese/t5-base.wwm/500000
fi

SAVE_PATH="${WORKING_DIR}/checkpoints/${DATASETS}/${MODEL}/ep${EPOCHS}_bs${BATCH_SIZE}_lr${LR}_G${GRAD_ACC}"

OPTS=""
OPTS+=" --model_name_or_path ${CKPT_PATH}"
OPTS+=" --data_path ${WORKING_DIR}/dataset/"
OPTS+=" --datasets ${DATASETS}"
OPTS+=" --output_dir ${SAVE_PATH}"
OPTS+=" --max_source_length ${MAX_SRC_LEN}"
OPTS+=" --max_target_length ${MAX_TGT_LEN}"
OPTS+=" --val_max_target_length ${MAX_TGT_LEN}"
OPTS+=" --do_train"
# OPTS+=" --max_train_samples 128"
OPTS+=" --do_eval"
# OPTS+=" --max_eval_samples 3"
OPTS+=" --do_predict"
# OPTS+=" --max_predict_samples 128"
OPTS+=" --num_train_epochs ${EPOCHS}"
# OPTS+=" --max_steps ${STEPS}"
OPTS+=" --per_device_train_batch_size ${BATCH_SIZE}"
OPTS+=" --per_device_eval_batch_size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient_accumulation_steps ${GRAD_ACC}"
OPTS+=" --learning_rate ${LR}"
OPTS+=" --logging_steps 100"
OPTS+=" --evaluation_strategy epoch"
# OPTS+=" --eval_steps 5000"
# OPTS+=" --save_strategy steps"
# OPTS+=" --save_steps 5000"
# OPTS+=" --warmup_steps 10000"
OPTS+=" --warmup_ratio 0.1"
OPTS+=" --disable_tqdm True"
OPTS+=" --load_best_model_at_end True"
OPTS+=" --metric_for_best_model ${METRIC}"

OPTS+=" --save_total_limit 3"
OPTS+=" --logging_dir ${SAVE_PATH}/runs"
OPTS+=" --fp16"
OPTS+=" --num_beams 4"
OPTS+=" --predict_with_generate"
# OPTS+=" --temperature ${TEMPERATURE}"
OPTS+=" --model_type ${MODEL_TYPE}"

if [[ $MODEL_TYPE == "pathway" ]]; then
    OPTS+=" --topk_experts -1"
fi

if [[ $SETTING == "skill" ]]; then
    OPTS+=" --expert_ids ${EXPERT_IDS}"
fi

# OPTS+=" --overwrite_cache True"


# Use DataParallel
# CMD="python3 -u ${WORKING_DIR}/run/run_multitask.py ${OPTS}"

# Use DistributedDataParallel
my_port=23456
# 1. Single-Node multi-process distributed training
CMD="python3 -u -m torch.distributed.launch --nproc_per_node=8  ${WORKING_DIR}/run/run_multitask.py ${OPTS}"
# 2. Multi-Node multi-process distributed training
# CMD="python3 -u -m torch.distributed.launch --nnodes=$HOST_NUM --node_rank=$INDEX --nproc_per_node $HOST_GPU_NUM --master_addr $CHIEF_IP --master_port $my_port ${WORKING_DIR}/run/run_multitask.py ${OPTS}"

echo ${CMD}
mkdir -p ${SAVE_PATH}

# export CUDA_VISIBLE_DEVICES=$CUDA
# export NCCL_IB_DISABLE=1
${CMD} 2>&1 | tee ${SAVE_PATH}/train.log

if [[ $SETTING =~ "ckpt" || $SETTING == "hs" ]]; then
    exit
fi

~/cache.sh
