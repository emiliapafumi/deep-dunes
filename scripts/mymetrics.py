import tensorflow as tf

class FScore(tf.keras.metrics.Metric):
    """
    Custom metric for F1-Score.

    Code borrowed from: 
    https://stackoverflow.com/questions/64474463/custom-f1-score-metric-in-tensorflow
    """

    def __init__(self, class_id, name=None, **kwargs):
        if not name:
            name = f'f_score_{class_id}'
        super().__init__(name=name, **kwargs)
        self.f1 = self.add_weight(name='f1', initializer='zeros')
        self.precision_fn = tf.keras.metrics.Precision(class_id=class_id)
        self.recall_fn = tf.keras.metrics.Recall(class_id=class_id)

    def update_state(self, y_true, y_pred, sample_weight=None):
        p = self.precision_fn(y_true, y_pred)
        r = self.recall_fn(y_true, y_pred)
        # since f1 is a variable, we use assign
        self.f1.assign(2 * ((p * r) / (p + r + 1e-6)))

    def result(self):
        return self.f1

    def reset_state(self):
        # we also need to reset the state of the precision and recall objects
        self.precision_fn.reset_states()
        self.recall_fn.reset_states()
        self.f1.assign(0)
