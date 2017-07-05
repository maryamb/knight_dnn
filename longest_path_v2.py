import tensorflow as tf

"""
Improved by:
  1. Added tf.HParams
  2. Added tf.Print
  3. Used MonitoredTrainingSession
  4. Added summaries
  5. Added a single funciton run_train
"""
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')

def hyper_params():
  return tf.contrib.training.HParams(batch_size=500, layer_sizes=[31, 19, 1],
          learning_rate=0.001, steps=30000, board_size=8,
          save_summaries_steps=100, save_checkpoint_secs=10)


def is_valid(moves, board_size):
  a = moves[:, :-1]
  b = moves[:, 1:]
  c = tf.abs(b - a)
  d = tf.div(c, board_size)
  e = tf.mod(c, board_size)
  all_ones = tf.ones_like(e)
  all_twos = 2 * all_ones
  move_check_1 = tf.logical_and(tf.equal(e, all_ones), tf.equal(d, all_twos))
  move_check_2 = tf.logical_and(tf.equal(e, all_twos), tf.equal(d, all_ones))
  move_check = tf.reduce_all(tf.logical_or(move_check_1, move_check_2))
  moves_unpack = tf.unstack(moves)
  is_not_repeated = tf.stack([tf.cond(tf.equal(tf.size(tf.unique(row)[0]),
      tf.size(row)), lambda: True, lambda: False) for row in moves_unpack])
  location_check = tf.reduce_all(tf.logical_and(tf.greater_equal(moves, 0),
      tf.less(moves, board_size * board_size)))
  return tf.to_float(tf.logical_and(move_check,
      tf.logical_and(location_check, is_not_repeated)))


def build_model(layer_sizes, input_tensor, label_tensor):
  layer = tf.contrib.layers.fully_connected(
        inputs=tf.to_float(input_tensor), activation_fn=tf.nn.relu,
        num_outputs=layer_sizes[0])
  layer = tf.contrib.layers.fully_connected(
        inputs=layer, activation_fn=tf.nn.relu, num_outputs=layer_sizes[1])
  logits = tf.contrib.layers.fully_connected(
        inputs=layer, activation_fn=tf.nn.relu, num_outputs=layer_sizes[2])
  probabilities = tf.contrib.layers.fully_connected(
        inputs=logits, activation_fn=tf.sigmoid, num_outputs=1)
  return logits, probabilities


def create_input(board_size, batch_size):
  with tf.name_scope("input_layer"):
    r = tf.range(board_size * board_size)
    random_inp = tf.stack([tf.random_shuffle(r) for _ in range(batch_size)])
    random_input = tf.Print(random_inp, [random_inp], first_n=1, summarize=50)
    valid = tf.stack(is_valid(random_inp, board_size))
    valid = tf.reshape(valid, [batch_size, 1])
    return random_inp, valid


def run_train(train_dir):
  hps = hyper_params()
  features, labels = create_input(hps.board_size, hps.batch_size)
  logits, probabilities = build_model(hps.layer_sizes, features, labels)
  loss = tf.losses.sigmoid_cross_entropy(labels, logits)
  optimizer = tf.train.AdamOptimizer(hps.learning_rate)
  global_step = tf.contrib.framework.get_or_create_global_step()
  train_op = optimizer.minimize(loss, global_step=global_step)
  tf.summary.scalar("loss", loss)
  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=train_dir,
      save_checkpoint_secs=hps.save_checkpoint_secs,
      save_summaries_steps=hps.save_summaries_steps) as sess:
    continue_training = True
    while continue_training:
      f, l, t, out_loss = sess.run(fetches=[features, labels, train_op, loss])
      out_global_step = sess.run(global_step)
      if out_global_step % 100 == 0:
        print ("Global step: %d, Loss: %f" % (out_global_step, out_loss))
      if out_global_step > hps.steps:
        continue_training = False


def main():
  run_train(FLAGS.train_dir) 


if __name__ == "__main__":
  main()
