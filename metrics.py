from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

def get_cm(y_true, y_pred, labels=None, return_mat=False):
  cm = confusion_matrix(y_true, y_pred, labels=labels)
  return cm if return_mat else cm.ravel()

def precision(cm=None, y_true=None, y_pred=None, labels=None):
  if cm is not None and y_true is None and y_pred is None:
    tn, fp, fn, tp = cm
  else:
    tn, fp, fn, tp = get_cm(y_true, y_pred, labels)
  return tp / (tp + fp)

def recall(cm=None, y_true=None, y_pred=None, labels=None):
  if cm is not None and y_true is None and y_pred is None:
    tn, fp, fn, tp = cm
  else:
    tn, fp, fn, tp = get_cm(y_true, y_pred, labels)
  return tp / (tp + fn)

def specificity(cm=None, y_true=None, y_pred=None, labels=None):
  if cm is not None and y_true is None and y_pred is None:
    tn, fp, fn, tp = cm
  else:
    tn, fp, fn, tp = get_cm(y_true, y_pred, labels)
  return tn / (tn + fp)

def accuracy(cm=None, y_true=None, y_pred=None, labels=None):
  if cm is not None and y_true is None and y_pred is None:
    tn, fp, fn, tp = cm
  else:
    tn, fp, fn, tp = get_cm(y_true, y_pred, labels)
  return (tp + tn) / (tp + tn + fp + fn)

def f_score(cm=None, y_true=None, y_pred=None, labels=None, beta=1):
  if cm is not None and y_true is None and y_pred is None:
    tn, fp, fn, tp = cm
  else:
    tn, fp, fn, tp = get_cm(y_true, y_pred, labels)
  bso = (beta**2 + 1)
  return (bso*tp) / (bso*tp + beta**2*fn + fp)

def roc(y_true, y_pred):
  fpr, tpr, thresholds = roc_curve(y_true, y_pred, drop_intermediate=False)
  auc = roc_auc_score(y_true, y_pred)
  return fpr, tpr, auc, thresholds
