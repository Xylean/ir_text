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
myIndex.construct(idf = True)

#print(myIndex.index.inverted['eat'])
'''
print('\n- Searching with Dice :')
start = time.time()

for query in queries['queries']:
    myIndex.search(query)
'''
print('\n- Searching with TF :')
start = time.time()
res = myIndex.search(queries['queries'][0], ir_text.Measures.TF)[:5]
print('Results in', time.time() - start,  's :', res)
'''
print('\n- Searching with TF_IDF :')
start = time.time()
print('Results in', time.time() - start,  's :', myIndex.search(queries['queries'][0], ir_text.Measures.TF_IDF)[:5])
'''
