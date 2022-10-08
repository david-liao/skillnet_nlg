import sys
import re
import os
import json
import subprocess
import time

model_output_path = sys.argv[1]
env_node_rank = int(os.environ.get("INDEX", -1))
print(f"****** Node rank: {env_node_rank}  ******")

_re_checkpoint = re.compile(r"checkpoint\-(\d+)$")

content = os.listdir(model_output_path)
checkpoints = [
    os.path.join(model_output_path, path)
    for path in content
    if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(model_output_path, path))
]

checkpoints.sort(key=lambda x: int(_re_checkpoint.search(x).groups()[0]))


task_list = [
    ('lcsts',   'rouge-l'   ),
    ('adgen',   'bleu'      ),
    ('matinf',  'rouge-l'   ),
    ('kdconv',  'bleu'      ),
    ('nlpcc',   'sentence_bleu'),
]

# Evaluate on all checkpoints for all task
ckpt_tasks_dict = {}
for checkpoint in checkpoints:
    ckpt_tasks_dict[checkpoint] = {}
    for task, metric in task_list:
        evaluate_results_file = os.path.join(checkpoint, 'eval', task, 'evaluate_results.json')
        if not os.path.exists(evaluate_results_file):
            print(f"*** Evalute for {task} on {checkpoint}  ***")
            cmd = f"bash scripts/eval_predict_dist.sh    eval    {task}   {checkpoint}   16   512   200     4    {metric}"
            print(cmd)
            os.system(cmd)

        if env_node_rank != 0:
            while not os.path.exists(evaluate_results_file):
                time.sleep(1)

        with open(evaluate_results_file, 'r', encoding='utf-8') as f:
            evaluate_results = json.load(f)
            ckpt_tasks_dict[checkpoint][task] = evaluate_results[f"evaluate_{metric}"]


# output all evaluation results to file
if env_node_rank == 0:
    with open(os.path.join(model_output_path, 'all_evaluate_results.json'), 'w', encoding='utf-8') as f:
        json.dump(ckpt_tasks_dict, f, indent=4)


best_checkpoint, best_scores = max(ckpt_tasks_dict.items(), key=lambda x: sum(x[1].values()))
print("*** Best validation checkpoint and scores ***")
print(best_checkpoint, best_scores)

def extract_results(predict_results_file):
    line = subprocess.getoutput(f"cat {predict_results_file} | grep '^predict :'")
    prefix_len = len("predict : ")
    line = line[prefix_len:]
    result = json.loads(line.strip())
    return result


def extract_results_for_nlpcc(predict_results_file):
    prefix_len = len("Precision   : ")

    def extract_metric(metric):
        line = subprocess.getoutput(f"cat {predict_results_file} | grep '^{metric}'")
        line = line[prefix_len:]
        return float(line.strip()) * 100

    precision = extract_metric('Precision')
    recall = extract_metric('Recall')
    f_0_5 = extract_metric('F_0.5')
    return precision, recall, f_0_5


def extract_results_from_file(task, metric, predict_results_file, task_metrics_dict):
    if task in ('adgen', 'lcsts', 'kdconv'):
        result = extract_results(predict_results_file)
        task_metrics_dict[task] = {
            metric: result[metric],
        }
    elif task in ('matinf'):
        result = extract_results(predict_results_file)
        task_metrics_dict[task] = {
            'rouge-1': result['rouge-1'],
            'rouge-2': result['rouge-2'],
            'rouge-l': result['rouge-l'],
        }
    elif task in ('nlpcc'):
        precision, recall, f_0_5 = extract_results_for_nlpcc(predict_results_file)
        task_metrics_dict[task] = {
            'precision' : precision,
            'recall'    : recall,
            'f_0_5'     : f_0_5,
        }


# Predict on best checkpoint for all task
task_metrics_dict = {}
for task, metric in task_list:
    predict_metrics_file = os.path.join(best_checkpoint, 'predict', task, 'predict_metrics.json')
    if not os.path.exists(predict_metrics_file):
        print(f"*** Predict for {task} on {best_checkpoint}  ***")
        cmd = f"bash scripts/eval_predict_dist.sh    predict    {task}   {best_checkpoint}   16   512   200     4    {metric}"
        print(cmd)
        os.system(cmd)

    predict_results_file = os.path.join(best_checkpoint, 'predict', task, 'predict_results.txt')
    if env_node_rank == 0 and not os.path.exists(predict_results_file):
        print(f"*** Compute prediction metric scores for {task} from {predict_metrics_file}  ***")
        if task in ('adgen', 'lcsts', 'kdconv'):
            os.system(f"python3 -u scripts/eval/run_eval_metrics_bert.py dataset/{task}/test.jsonl {predict_metrics_file} 2>&1 | tee {predict_results_file}")
        elif task in ('matinf'):
            os.system(f"python3 -u scripts/eval/run_eval_metrics_matinf_ljw.py dataset/{task}/test.jsonl {predict_metrics_file} 2>&1 | tee {predict_results_file}")
        elif task in ('nlpcc'):
            os.system(f"bash scripts/eval/run_eval_metrics_nlpcc.sh {predict_metrics_file} 2>&1 | tee {predict_results_file}")
        else:
            print(f"The {task} is not supported!")
            exit()
    
    print(f"*** Extract prediction metric scores for {task} from {predict_results_file}  ***")
    if env_node_rank == 0:
        extract_results_from_file(task, metric, predict_results_file, task_metrics_dict)
    else:
        while True:
            try:
                extract_results_from_file(task, metric, predict_results_file, task_metrics_dict)
            except Exception as e:
                time.sleep(1)
                continue
            break


# output the final result according the experimental table format
def output_final_result(task_metrics_dict):
    dataset_col, metric_col, scores_col = [], [], []
    for task, _ in task_list:
        dataset_col.extend([task] * len(task_metrics_dict[task]))
        metric_col.extend(list(task_metrics_dict[task]))
        scores_col.extend(list(task_metrics_dict[task].values()))


    dataset_str = '\t'.join(dataset_col) 
    metric_str = '\t'.join(map(lambda x: x[:7], metric_col))
    score_str = '\t'.join(map(lambda x: str(round(x, 2)), scores_col))

    score_file_for_excel = "./results/score.txt"
    with open(score_file_for_excel, 'a', encoding='utf-8') as f:
        f.write(best_checkpoint + '\n')
        f.write(dataset_str + '\n')
        f.write(metric_str + '\n')
        f.write(score_str + '\n\n')

    print('\n')
    print(best_checkpoint)
    print(dataset_str)
    print(metric_str)
    print(score_str)

if env_node_rank == 0:
    output_final_result(task_metrics_dict)

os.system("~/cache.sh")
