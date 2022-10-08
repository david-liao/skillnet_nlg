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

    # solve the reporting error for rouge call.
    def invalid_hyp(hyp):
        return len([" ".join(_.split()) for _ in hyp.split(".") if len(_) > 0]) <= 0
    rouge_preds = [*map(lambda x: '' if invalid_hyp(x) else x, preds)]  # cannot be '' after split by "."
    rouge_score = rouge.get_scores(rouge_preds, refs, avg=True, ignore_empty=True)  # hyp can be '' when ignore_empty is set True

    # rouge_score = rouge.get_scores(preds, refs, avg=True)
    rouge_score = {key: value['f'] * 100 for key, value in rouge_score.items()}
    score.update(rouge_score)

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

preds.pop(40736)
preds.pop(93604)
preds.pop(173494)

# for ref
ref_path = sys.argv[1]
print('ref_path: ', ref_path)

lines = open(ref_path, encoding="utf-8").readlines()
raw_refs = [json.loads(line)["summary"].strip() for line in lines]
refs = [convert(ref) for ref in raw_refs]

print('test example:')
print(raw_refs[0])
print(refs[0])

refs.pop(40736)
refs.pop(93604)
refs.pop(173494)

assert len(preds) == len(refs)

score = compute_score(preds, refs)
print("predict :", json.dumps(score))