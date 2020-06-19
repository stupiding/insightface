import mxnet as mx

class AccMetric(mx.metric.EvalMetric):
  def __init__(self):
    self.axis = 1
    super(AccMetric, self).__init__(
        'acc', axis=self.axis,
        output_names=None, label_names=None)
    self.losses = []
    self.count = 0

  def update(self, labels_ls, preds_ls):
    self.count+=1
    labels = [labels_ls[0][:, i] for i in range(len(preds_ls) - 1)] if len(preds_ls) > 2 else labels_ls
    for label, pred_label in zip(labels, preds_ls[1:]):
        if pred_label.shape != label.shape:
            pred_label = mx.ndarray.argmax(pred_label, axis=self.axis)
        pred_label = pred_label.asnumpy().astype('int32').flatten()
        label = label.asnumpy()
        if label.ndim==2:
            label = label[:,0]
        label = label.astype('int32').flatten()
        assert label.shape==pred_label.shape
        pred_label, label = pred_label.flat, label.flat
        #valid_ids = np.argwhere(label.asnumpy() != -1)
        self.sum_metric += (pred_label == label).sum()
        self.num_inst += len(pred_label)

class LossValueMetric(mx.metric.EvalMetric):
  def __init__(self):
    self.axis = 1
    super(LossValueMetric, self).__init__(
        'lossvalue', axis=self.axis,
        output_names=None, label_names=None)
    self.losses = []

  def update(self, labels, preds):
    print(labels[0].shape, preds[0].shape)
    loss = preds[-1].asnumpy()[0]
    self.sum_metric += loss
    self.num_inst += 1.0
    gt_label = preds[-2].asnumpy()
    #print(gt_label)
