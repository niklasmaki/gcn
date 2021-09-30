import time

import numpy as np
import tensorflow as tf

from gcn.utils import *

def evaluate(sess, model, features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


def get_soft_prediction_labels(sess, model, features, support, labels, mask, placeholders):
  feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
  feed_dict_val.update({placeholders['dropout']: 0})
  return sess.run(tf.nn.softmax(model.outputs), feed_dict=feed_dict_val)


def train_model(FLAGS, sess, model, features, support, y_train, y_val, train_mask, val_mask, placeholders):
  cost_val = []
  final_pred_soft = 0
  final_pred = 0
  for epoch in range(FLAGS.epochs):
      t = time.time()
      # Construct feed dictionary
      feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
      feed_dict.update({placeholders['dropout']: FLAGS.dropout})

      # Training step
      outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

      # Validation
      cost, acc, duration = evaluate(sess, model, features, support, y_val, val_mask, placeholders)
      cost_val.append(cost)

      # Print results
      print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
            "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
            "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

      if epoch > FLAGS.epoch_to_start_collect_weights:
        labels = get_soft_prediction_labels(sess, model, features, support, y_val, val_mask, placeholders)
        final_pred_soft += labels
        final_pred = final_pred_soft.argmax(axis=1)

      if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
          print("Early stopping...")
          break
  
  return final_pred_soft, final_pred