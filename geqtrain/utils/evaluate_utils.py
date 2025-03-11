from sklearn.metrics import ( # TODO can I bring imports inside AccuracyMetric.__init__
  precision_recall_fscore_support,
  confusion_matrix,
  roc_auc_score,
  balanced_accuracy_score,
  f1_score,
  precision_recall_curve,
  auc
)

def print_eval(gt, preds, logits):

#   if len(logits)<5:
#       # single smile eval mode
#       for i in range(len(logits)): print(f"gt: {gt[i]}, preds: {preds[i]}, logits: {logits[i]}")
#       return

  cm = confusion_matrix(gt, preds, labels=[False, True])
  tn, fp, fn, tp = cm.ravel()
  matrix_string = (
      f"Confusion Matrix:\n"
      f"                Predicted\n"
      f"                Positive     Negative\n"
      f"Actual Positive   TP: {tp}        FN: {fn}\n"
      f"       Negative   FP: {fp}        TN: {tn}"
  )
  print(matrix_string)
  print(f"Correctly classified: {tn+tp}, misclassified: {fp+fn}")
  _roc_auc_score = roc_auc_score(gt, logits)
  print('_roc_auc_score: ', _roc_auc_score)
  _precision, _recall, thresholds = precision_recall_curve(gt, logits)
  print('_precision_recall_curve: ', auc(_recall, _precision))
  precision, recall, fscore, support = precision_recall_fscore_support(gt, preds)
  print('precision: ', precision)
  print('recall: ', recall)
  print('fscore: ', fscore)
  print('support: ', support)
  ba = balanced_accuracy_score(gt, preds)
  f1 = f1_score(gt, preds, average='binary')
  print('balanced accuracy: ',ba)
  print('f1 score: ', f1)
  print('\n')


class AccuracyMetric(object):
    def __init__(self, key:str):
        '''key: key to be taken from ref_data'''
        self.key = key
        self._gts_list, self._preds_list, self._logits_list = [], [], []

    def __call__(self, batch_index, chunk_index, out, ref_data, data, pbar, **kwargs):
        '''the signature of this method must always match: out_callback signature in evaluate.py'''
        target = ref_data[self.key].cpu().bool()
        _logits = out[self.key].sigmoid()
        prediction = (_logits>.5).cpu().bool()

        target.squeeze().dim()
        self._gts_list += target.squeeze().tolist() if target.squeeze().dim() != 0 else [target.squeeze().tolist()]
        self._logits_list += _logits.squeeze().tolist() if _logits.squeeze().dim() != 0 else [_logits.squeeze().tolist()]
        self._preds_list += prediction.squeeze().tolist() if prediction.squeeze().dim() != 0 else [prediction.squeeze().tolist()]

    def print_current_result(self):
        print_eval(self._gts_list, self._preds_list, self._logits_list)

