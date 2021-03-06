# tfmetrics
package "tfmetrics" will be used to eval f1_score, recall_score, precision_score in tensorflow>1.12

# Usage

```python
from sklearn.metrics import recall_score
import tensorflow as tf
from tfmetrics.metrics import evalsequence
y_true = tf.constant([[0, 0, 0, 1, 2, 2, 0], [3, 4, 0, 0, 0, 0, 0]])
y_pred = tf.constant([[0, 0, 1, 2, 2, 2, 0], [3, 4, 0, 0, 0, 0, 0]])

#compare with sklaern metrics
y_true_ = [0, 0, 0, 1, 2, 2, 0, 3, 4, 0, 0, 0, 0, 0]
y_pred_ = [0, 0, 1, 2, 2, 2, 0, 3, 4, 0, 0, 0, 0, 0]
recall = recall_score(y_true_, y_pred_, average="macro")
eval_result = evalsequence(y_true, y_pred, 5)

print("Recall score by 'tfmetrics' {:.4f}".format(eval_result.recall_score))
print("Recall score by 'sklearn' {:.4f}".format(recall))

```

# Installation

```bash
python setup.py tfmetrics

```

# Requirement
Tensorflow >= 1.12