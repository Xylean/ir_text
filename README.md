# ir_text
A simple Information Research python package

## Simple Demo

```python
# data and queries should be {'dataset' : lists -> [{'text' : [...], 'id' = int}]}
an_inverted_index = ir_text.InvertedIndex(data['dataset'], language = 'english')
a_linear_index = ir_text.LinearIndex(data['dataset'], language = 'english')

a_linear_index.construct()
an_inverted_index.construct(idf = True)

results_linear = a_linear_index.search(queries['queries'][0])
results_inverted = an_inverted_index.search(queries['queries'][0], ir_text.Measures.TF)
```
A similar but more complete example is available in ```ir_text_notebook.ipynb```.

### To Do:
- [ ] Specify stoplist path
- [x] Finish notebook
- [x] Linear index evaluation
- [x] Inverted index evaluation
