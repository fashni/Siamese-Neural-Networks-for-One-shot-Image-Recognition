from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

def get_cm(y_true, y_pred, labels=None, return_mat=False):
  cm = confusion_matrix(y_true, y_pred, labels=labels)
  return cm if return_mat else cm.ravel()

def precision(y_true, y_pred, labels=None):
  tn, fp, fn, tp = get_cm(y_true, y_pred, labels)
  return tp / (tp + fp)

def recall(y_true, y_pred, labels=None):
  tn, fp, fn, tp = get_cm(y_true, y_pred, labels)
  return tp / (tp + fn)

def specificity(y_true, y_pred, labels=None):
  tn, fp, fn, tp = get_cm(y_true, y_pred, labels)
  return tn / (tn + fp)

def accuracy(y_true, y_pred, labels=None):
  tn, fp, fn, tp = get_cm(y_true, y_pred, labels)
  return (tp + tn) / (tp + tn + fp + fn)

def f_score(y_true, y_pred, labels=None, beta=1):
  tn, fp, fn, tp = get_cm(y_true, y_pred, labels)
  bso = (beta**2 + 1)
  return (bso*tp) / (bso*tp + beta**2*fn + fp)

def roc(y_true, y_pred):
  fpr, tpr, _ = roc_curve(y_true, y_pred, drop_intermediate=False)
  auc = roc_auc_score(y_true, y_pred)
  return fpr, tpr, auc
