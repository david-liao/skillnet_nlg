# predict on specific checkpoint and print score
import sys
import re
import os
import json
import subprocess
import time

env_node_rank = int(os.environ.get("INDEX", -1))
print(f"****** Node rank: {env_node_rank}  ******")

task_list = [
    ('lcsts',   'rouge-l'   ),
    ('adgen',   'bleu'      ),
    ('matinf',  'rouge-l'   ),
    ('kdconv',  'bleu'      ),
    ('nlpcc',   'sentence_bleu'),
]


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


# Predict on denoted checkpoint for all task
def predict_on_checkpoint(checkpoint):
    task_metrics_dict = {}
    for task, metric in task_list:
        predict_metrics_file = os.path.join(checkpoint, 'predict', task, 'predict_metrics.json')
        if not os.path.exists(predict_metrics_file):
            print(f"*** Predict for {task} on {checkpoint}  ***")
            cmd = f"bash scripts/eval_predict_dist.sh    predict    {task}   {checkpoint}   16   512   200     4    {metric}"
            print(cmd)
            os.system(cmd)

        predict_results_file = os.path.join(checkpoint, 'predict', task, 'predict_results.txt')
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

    return task_metrics_dict

# output the final result according the experimental table format
def output_final_result(task_metrics_dict, checkpoint):
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
        f.write(checkpoint + '\n')
        f.write(dataset_str + '\n')
        f.write(metric_str + '\n')
        f.write(score_str + '\n\n')

    print('\n')
    print(checkpoint)
    print(dataset_str)
    print(metric_str)
    print(score_str)

def main():
    predict_checkpoint = sys.argv[1]
    task_metrics_dict = predict_on_checkpoint(predict_checkpoint)

    if env_node_rank == 0:
        output_final_result(task_metrics_dict, predict_checkpoint)

    os.system("~/cache.sh")    


if __name__ == '__main__':
    main()
