import json
import sys
import os
import time
from multiprocessing import Process
import os
import time
# from more_itertools import divide

file = sys.argv[1]

data = json.load(open(file, 'r', encoding='utf-8'))
if 'outputs' in data:
    hyps = [pred['prediction'].strip() for pred in data['outputs']]
elif 'predictions' in data:
    hyps = [pred.strip() for pred in data['predictions']]

dir = './tmp/'
for path in os.listdir(dir):
    os.remove(dir + path)

print(os.listdir(dir))

with open('./tmp/output.txt', 'w') as f:
    f.write('\n'.join(hyps))

# 去除空格（分词信息）
cmd1 = "sed 's/ //g' ./tmp/output.txt > ./tmp/output.txt.remove.spac"
print(cmd1)
os.system(cmd1)

# 使用pkunlp进行分词
cmd2 = "python3 pkunlp_segment.py --corpus ./tmp/output.txt.remove.spac --segsuffix seg"
print(cmd2)
os.system(cmd2)

# 使用m2score计算得分
# cmd3 = "./m2scorer/m2scorer ./tmp/output.txt.remove.spac.seg  ./gold.01"
cmd3 = "python3 ./m2scorer/scripts/m2scorer.py ./tmp/output.txt.remove.spac.seg  ./gold.01"
print(cmd3)
os.system(cmd3)
