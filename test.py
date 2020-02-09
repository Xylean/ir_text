import json
import ir_text
import time

with open('data/datasets/cacm_dataset.json', 'r') as json_data:
    data = json.load(json_data)
with open('data/datasets/cacm_queries.json', 'r') as json_queries:
    queries = json.load(json_queries)

print('Creating Index ...')
myIndex = ir_text.InvertedIndex(data['dataset'])
print('Constructing Index ...')
myIndex.construct()

print('\n- Searching with Dice :')
start = time.time()
for query in queries['queries']:
    myIndex.search(query)
print('Results in', time.time() - start,  's :', res)

print('\n- Searching with TF :')
start = time.time()
for query in queries['queries']:
    myIndex.search(query, ir_text.Measures.TF)
print('Results in', time.time() - start,  's :', res)
