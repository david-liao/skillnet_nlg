import sys
import json
import numpy as np

# argv[1]: task_name, argv[2]: reference (label), argv[3]: hypothesis (prediction)
assert len(sys.argv) == 4

# bert tokenizer
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('scripts/eval')
print(tokenizer.decode([99]))
print(tokenizer)

def convert(text):
    return tokenizer.decode(tokenizer.encode(text.strip()), skip_special_tokens=True).strip()


# compute score
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction, sentence_bleu
from rouge import Rouge
import jieba
rouge = Rouge()

def bleu(data):
    """
    compute rouge score
    Args:
        data (list of dict including reference and candidate):
    Returns:
            res (dict of list of scores): rouge score
    """

    res = {}
    for i in range(1, 5):
        res["bleu-%d"%i] = []

    for tmp_data in data:
        origin_candidate = tmp_data['candidate']
        origin_reference = tmp_data['reference']
        assert isinstance(origin_candidate, str)
        if not isinstance(origin_reference, list):
            origin_reference = [origin_reference]

        for i in range(1, 5):
            res["bleu-%d"%i].append(sentence_bleu(references=[r.strip().split() for r in origin_reference], hypothesis=origin_candidate.strip().split(), weights=tuple([1./i for j in range(i)]), smoothing_function=SmoothingFunction().method3)) 

    for key in res:
        res[key] = np.mean(res[key])
        
    return res

def proline(line):
    return " ".join([w for w in jieba.cut("".join(line.strip().split()))])

def compute_score(task, preds, refs):
    score = {}

    assert '' not in preds
    # while '' in preds:
    #     idx=preds.index('')
    #     preds[idx]='。'

    rouge_score = rouge.get_scores(preds, refs, avg=True)
    rouge_score = {key: value['f'] * 100 for key, value in rouge_score.items()}
    score.update(rouge_score)

    # acc = sum([int(pred == ref) for pred, ref in zip(preds, refs)]) / len(preds)
    # score['acc'] = acc

    preds_lst = [pred.split() for pred in preds]
    refs_lst = [[ref.split()] for ref in refs]

    if task in ('t2e'):
        # eval_data = [{"reference": ref, "candidate": pred} for ref, pred in zip(refs, preds)]
        # bleu_score = bleu(eval_data)
        # score.update(bleu_score)
        for i in range(1,5):
            weights = (1./i,) * i
            bleu_score = corpus_bleu(refs_lst, preds_lst, weights=weights, smoothing_function=SmoothingFunction().method3)
            score[f'bleu-{i}'] = bleu_score * 100
    elif task in ('outgen'):
        eval_data = [{"reference": proline(ref), "candidate": proline(pred)} for ref, pred in zip(refs, preds)]
        print('evaluation example:')
        print(eval_data[0])
        bleu_score = bleu(eval_data)
        bleu_score = {key: value * 100 for key, value in bleu_score.items()}
        score.update(bleu_score)
    else:
        try:
            bleu_score = corpus_bleu(refs_lst, preds_lst, smoothing_function=SmoothingFunction().method3)
        except ZeroDivisionError as _:
            bleu_score = 0
        score['bleu'] = bleu_score * 100

    return score


# for pred
pred_path = sys.argv[3]
print('pred_path: ', pred_path)

data = json.load(open(pred_path, encoding="utf-8"))
if 'outputs' in data:
    raw_preds = [pred['prediction'].strip() for pred in data['outputs']]
elif 'predictions' in data:
    raw_preds = [pred.strip() for pred in data['predictions']]
preds = [convert(pred) for pred in raw_preds]

print('pred example:')
print(raw_preds[0])
print(preds[0])


# for ref
ref_path = sys.argv[2]
print('ref_path: ', ref_path)

lines = open(ref_path, encoding="utf-8").readlines()
raw_refs = [json.loads(line)["summary"].strip() for line in lines]
refs = [convert(ref) for ref in raw_refs]

print('test example:')
print(raw_refs[0])
print(refs[0])

assert len(preds) == len(refs)

task = sys.argv[1]
score = compute_score(task, preds, refs)
print("predict :", json.dumps(score))