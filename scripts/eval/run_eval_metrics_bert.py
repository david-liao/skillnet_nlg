import sys
import json

# bert tokenizer
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('scripts/eval')
print(tokenizer.decode([99]))
print(tokenizer)

def convert(text):
    return tokenizer.decode(tokenizer.encode(text.strip()), skip_special_tokens=True).strip()


# compute score
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge import Rouge
rouge = Rouge()

def compute_score(preds, refs):
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
    try:
        bleu_score = corpus_bleu(refs_lst, preds_lst, smoothing_function=SmoothingFunction().method3)
    except ZeroDivisionError as _:
        bleu_score = 0

    score['bleu'] = bleu_score * 100

    return score


# for pred
pred_path = sys.argv[2]
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
ref_path = sys.argv[1]
print('ref_path: ', ref_path)

lines = open(ref_path, encoding="utf-8").readlines()
raw_refs = [json.loads(line)["summary"].strip() for line in lines]
refs = [convert(ref) for ref in raw_refs]

print('test example:')
print(raw_refs[0])
print(refs[0])

assert len(preds) == len(refs)

score = compute_score(preds, refs)
print("predict :", json.dumps(score))