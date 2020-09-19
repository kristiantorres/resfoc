import os
import tensorflow as tf

def cross_entropy_balanced(y_true, y_pred):
  y_true = tf.cast(y_true, tf.float32)

  count_neg = tf.reduce_sum(1. - y_true)
  count_pos = tf.reduce_sum(y_true)

  beta = count_neg / (count_neg + count_pos)

  pos_weight = beta / (1 - beta)

  cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)

  cost = tf.reduce_mean(cost * (1 - beta))

  return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)


os.environ['CUDA_VISIBLE_DEVICES'] = str(4)

ten1 = tf.constant([ 1, 1, 0, 1, 0,0 ])

ten2 = tf.constant([10.,30., 7., 11.2, 5.,15.])

loss = cross_entropy_balanced(ten1,ten2)

with tf.Session() as sess: print(loss.eval())
