# GNN-MCC
crisismmd

Explanation of key parameters:

`--use_agreed_label`：If `true`, perform either image-text pair classification or separate image/text classification when the image and text labels are consistent; if `false`, perform separate image and text classification when there is a label mismatch between the image and text.

`--predict_pairs`: If `true`, predict the image-text pair; If `false`, predict the image and text separately, and compute the overall prediction result.

`--task`: `informative` refers to the informativeness classification task, and `humanitarian` refers to the humanitarian classification task.

step 1:
```python
python crisismmd_graph.py
```
step 2: 
```python
python train.py
```

Contact Us: 1755330594@qq.com
