from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from gcn.utils import *
from gcn.models import GCN, MLP
from gcn.distances import node2vec_distances, neighborhood_distance_matrix, weighted_distance_matrix
from gcn.posterior import test, map_estimate, get_allowed_edges
from gcn.train_utils import train_model, evaluate
from gcn.metrics import masked_accuracy

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('epoch_to_start_collect_weights', 100, 'The epoch after which weights will be collected.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 1000, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

# Some preprocessing
features = preprocess_features(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
sess = tf.Session()

# Init variables
sess.run(tf.global_variables_initializer())

print("Calculating embeddings...")
d1 = node2vec_distances(adj, FLAGS.dataset)

train_model(FLAGS, sess, model, features, support, y_train, y_val, train_mask, val_mask, placeholders)
 
print("Base model optimization Finished!")

feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
preds = sess.run(model.outputs, feed_dict=feed_dict)

print("Calculating neighorhood distance matrix...")
d2 = neighborhood_distance_matrix(adj, preds, FLAGS.dataset)

print("Calculating weighted distance matrix...")
d = weighted_distance_matrix(d1, d2)

print("Calculating the MAP estimate...")
map_support = map_estimate(adj, d, FLAGS.dataset)
map_support = [preprocess_adj(map_support)]

print("Sampling weights by running a GCN with MC dropout...")

final_model = model_func(placeholders, input_dim=features[2][1], logging=True)
sess.run(tf.global_variables_initializer())
final_pred_soft, final_pred = train_model(FLAGS, sess, final_model, features, map_support, y_train, y_val, train_mask, val_mask, placeholders)

test_acc = sess.run(masked_accuracy(final_pred_soft, y_test, test_mask))
print("Test accuracy: ", "{:.5f}".format(test_acc))
