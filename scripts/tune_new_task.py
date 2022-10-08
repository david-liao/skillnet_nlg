import sys
import re
import os
import json
import subprocess
import time

WORKING_DIR='.'

# python scripts/tune_new_task.py   m2e   pathway.gpu8  "16"    "16 8 4 2 1"    "1e-5 3e-5 5e-5"
assert len(sys.argv) == 6

TASK=sys.argv[1]
MODEL=sys.argv[2]
EPOCHS=sys.argv[3]
BATCH_SIZE=sys.argv[4]
LR=sys.argv[5]

epochs = EPOCHS.split()
batches = BATCH_SIZE.split()
lrs = LR.split()

if TASK == "t2e":
    MAX_SRC_LEN=30
    MAX_TGT_LEN=170
    METRIC="bleu-2"
elif TASK == "pkupb":
    MAX_SRC_LEN=140
    MAX_TGT_LEN=140
    METRIC="bleu"
elif TASK == "outgen":
    MAX_SRC_LEN=100
    MAX_TGT_LEN=310
    METRIC="bleu-2"
else:
    print(f"Not supported task {TASK}")
    sys.exit()


score_dict = {}
for epoch in epochs:
    for batch in batches:
        for lr in lrs:
            config = f"ep{epoch}_bs{batch}_lr{lr}_G1"
            score_dict[config] = {}
            SAVE_PATH = f"{WORKING_DIR}/checkpoints/{TASK}/{MODEL}/{config}"
            predict_results_file = os.path.join(SAVE_PATH, 'predict_results.json')
            if not os.path.exists(predict_results_file):
                print(f"*** Finetune for {TASK} with (epoch: {epoch}, batch size: {batch}, lr: {lr})  ***")
                cmd = f"bash scripts/run_single_task.sh   {TASK}   {MODEL}   {epoch}   {batch}    {batch}   1   {lr}    {MAX_SRC_LEN}   {MAX_TGT_LEN}  {METRIC}"
                print(cmd)
                os.system(cmd)
            
            with open(predict_results_file, 'r', encoding='utf-8') as f:
                predict_results = json.load(f)
                if TASK == 'pkupb':
                    score_dict[config]['predict_bleu'] = predict_results['predict_bleu']
                elif TASK in ('t2e', 'outgen'):
                    score_dict[config]['predict_bleu-1'] = predict_results['predict_bleu-1']
                    score_dict[config]['predict_bleu-2'] = predict_results['predict_bleu-2']



# output all predict results to file
output_path = f"{WORKING_DIR}/checkpoints/{TASK}/{MODEL}"
with open(os.path.join(output_path, 'all_predict_results.json'), 'w', encoding='utf-8') as f:
    json.dump(score_dict, f, indent=4)


best_checkpoint, best_scores = max(score_dict.items(), key=lambda x: x[1][f'predict_{METRIC}'])
print("*** Best prediction checkpoint and scores ***")
print(best_checkpoint, best_scores)


os.system("~/cache.sh")
