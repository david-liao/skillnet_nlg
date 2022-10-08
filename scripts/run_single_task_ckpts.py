import sys
import re
import os
import json
import subprocess
import time

# python scripts/run_single_task_ckpts.py     t2e      pathway.gpu8.ckpt[.range_of_checkpoints]    16  16   16   1   3e-5     30   170  bleu-2   /apdcephfs/share_916081/duyu_shared_data/jwliao/pathway/checkpoints/adgen.kdconv.lcsts.matinf.nlpcc/pathway.gpu64.pretrain.500k/step100000_bs8_lr3e-5_G1_T4
assert len(sys.argv) == 12
WORKING_DIR='.'
DATASETS=sys.argv[1]
MODEL_PREFIX=sys.argv[2]
EPOCHS=sys.argv[3]
BATCH_SIZE=sys.argv[4]
EVAL_BATCH_SIZE=sys.argv[5]
GRAD_ACC=sys.argv[6]
LR=sys.argv[7]
MAX_SRC_LEN=sys.argv[8]
MAX_TGT_LEN=sys.argv[9]
METRIC=sys.argv[10]
model_path = sys.argv[11]

ckpt_range = None
if len(MODEL_PREFIX.split('.')) == 4:
    *others, ckpt_range = MODEL_PREFIX.split('.')
    begin, end = ckpt_range.split('_')
    MODEL_PREFIX = '.'.join(others)

_re_checkpoint = re.compile(r"checkpoint\-(\d+)$")

content = os.listdir(model_path)
checkpoints = [
    os.path.join(model_path, path)
    for path in content
    if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(model_path, path))
]

if ckpt_range:
    checkpoints = list(filter(lambda x: int(begin) <= int(_re_checkpoint.search(x).groups()[0]) <= int(end), checkpoints))

checkpoints.sort(key=lambda x: int(_re_checkpoint.search(x).groups()[0]))

print(f"*** Finetune for {DATASETS} on {len(checkpoints)} checkpoints:  ***")
for checkpoint in checkpoints:
    print(checkpoint)


# Evaluate on all checkpoints for all task
ckpt_tasks_dict = {}
for checkpoint in checkpoints:
    ckpt_tasks_dict[checkpoint] = {}
    ckpt_num = _re_checkpoint.search(checkpoint).groups()[0]
    MODEL_PATH = f"{MODEL_PREFIX}/{ckpt_num}"
    MODEL = f"{MODEL_PREFIX}.{ckpt_num}"
    SAVE_PATH = f"{WORKING_DIR}/checkpoints/{DATASETS}/{MODEL_PATH}/ep{EPOCHS}_bs{BATCH_SIZE}_lr{LR}_G{GRAD_ACC}"

    predict_results_file = os.path.join(SAVE_PATH, 'predict_results.json')
    if not os.path.exists(predict_results_file):
        print(f"*** Finetune for {DATASETS} on {checkpoint}  ***")
        cmd = f"bash scripts/run_single_task.sh   {DATASETS}   {MODEL}   {EPOCHS}   {BATCH_SIZE}    {EVAL_BATCH_SIZE}   {GRAD_ACC}   {LR}    {MAX_SRC_LEN}   {MAX_TGT_LEN}  {METRIC}  {checkpoint}"
        print(cmd)
        os.system(cmd)
    
    with open(predict_results_file, 'r', encoding='utf-8') as f:
        predict_results = json.load(f)
        if DATASETS == 'pkupb':
            ckpt_tasks_dict[checkpoint]['predict_bleu'] = predict_results['predict_bleu']
        elif DATASETS in ('t2e', 'outgen'):
            ckpt_tasks_dict[checkpoint]['predict_bleu-1'] = predict_results['predict_bleu-1']
            ckpt_tasks_dict[checkpoint]['predict_bleu-2'] = predict_results['predict_bleu-2']


# output all predict results to file
output_path = f"{WORKING_DIR}/checkpoints/{DATASETS}/{MODEL_PREFIX}"
if ckpt_range:
    file_name = f"all_predict_results.{begin}_{end}.json"
else:
    file_name = 'all_predict_results.json'
with open(os.path.join(output_path, file_name), 'w', encoding='utf-8') as f:
    json.dump(ckpt_tasks_dict, f, indent=4)


best_checkpoint, best_scores = max(ckpt_tasks_dict.items(), key=lambda x: x[1][f'predict_{METRIC}'])
print("*** Best prediction checkpoint and scores ***")
print(best_checkpoint, best_scores)


os.system("~/cache.sh")
