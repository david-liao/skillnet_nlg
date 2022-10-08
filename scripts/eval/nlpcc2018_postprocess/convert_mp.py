import json
import sys
import os
import time

nums = [str(i) for i in range(10)]
def rep(text):
    text = text.replace('...', '…').replace('?', '？').replace('!', '！').replace(';', '；').replace('(', '（').replace(')', '）')
    text = list(text)
    for i in range(len(text)):
        if i == 0 or i == len(text)-1:
            continue
        if text[i] == ',':
            if text[i-1] in nums and text[i+1] in nums:
                # print(text[i-1: i+2])
                continue
            else:
                text[i] = text[i].replace(',', '，')
        elif text[i] == ':':
            if text[i-1] in nums and text[i+1] in nums:
                # print(text[i-1: i+2])
                continue
            else:
                text[i] = text[i].replace(':', '：')
    return ''.join(text)

def get_score(file):
    print(file)
    data = json.load(open(file, 'r', encoding='utf-8'))
    if 'outputs' in data:
        hyps = [pred['prediction'].strip() for pred in data['outputs']]
    elif 'predictions' in data:
        hyps = [pred.strip() for pred in data['predictions']]

    hyps = [rep(hyp) for hyp in hyps]
    # hyps = [hyp for hyp in hyps]

    dir = './tmp/'
    for path in os.listdir(dir):
        if path.startswith('.'):
            continue
        os.remove(dir + path)

    print(os.listdir(dir))

    with open('./tmp/output.txt', 'w') as f:
        f.write('\n'.join(hyps))

    # 去除空格（分词信息）
    cmd1 = "sed 's/ //g' ./tmp/output.txt > ./tmp/output.txt.remove.spac"
    print(cmd1)
    os.system(cmd1)

    # 使用pkunlp进行分词
    cmd2 = "python pkunlp_segment.py --corpus ./tmp/output.txt.remove.spac --segsuffix seg"
    print(cmd2)
    os.system(cmd2)

    # 使用m2score计算得分
    cmd3 = "./m2scorer/m2scorer ./tmp/output.txt.remove.spac.seg  ./gold.01"
    print(cmd3)
    os.system(cmd3)


path = sys.argv[1]
print('pred from: ', path)

files = []
models = os.listdir(path)
for model in models:
    if any(m in model for m in ['bart-large', '5w', 'cpt']):
        continue
    exp_path = path + model
    if not os.path.isdir(exp_path):
        continue
    exps = os.listdir(exp_path)
    for exp in exps:
        file = os.path.join(exp_path, exp, 'predict_metrics.json')
        # print(file)
        if os.path.exists(file):
            files.append(file)

files.sort()
print('\n'.join(files))
print('total:', len(files))
# exit()

for file in files:
    get_score(file)
    # break

print("Done.")
