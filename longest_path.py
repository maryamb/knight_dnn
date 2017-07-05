import tensorflow as tf


_SIZE = 8
_BATCH_SIZE = 300
_STEPS = 30000


def is_valid(moves, size):
  a = moves[:, :-1]
  b = moves[:, 1:]
  c = tf.abs(b - a)
  d = tf.div(c, size)
  e = tf.mod(c, size)
  all_ones = tf.ones_like(e)
  all_twos = 2 * all_ones
  move_check_1 = tf.logical_and(tf.equal(e, all_ones), tf.equal(d, all_twos))
  move_check_2 = tf.logical_and(tf.equal(e, all_twos), tf.equal(d, all_ones))
  move_check = tf.reduce_all(tf.logical_or(move_check_1, move_check_2))
  moves_unpack = tf.unstack(moves)
  is_not_repeated = tf.stack([tf.cond(tf.equal(tf.size(tf.unique(row)[0]),
      tf.size(row)), lambda: True, lambda: False) for row in moves_unpack])
  location_check = tf.reduce_all(tf.logical_and(tf.greater_equal(moves, 0),
      tf.less(moves, size * size)))
  return tf.to_float(tf.logical_and(move_check,
      tf.logical_and(location_check, is_not_repeated)))

def to_index(row, col, size):
  return row * size + col

def build_model(input_tensor, label_tensor):
  LAYER_1_SIZE = 10
  LAYER_2_SIZE = 20
  LAYER_3_SIZE = 1
  layer = tf.contrib.layers.fully_connected(
        inputs=tf.to_float(input_tensor), activation_fn=tf.nn.relu, num_outputs=LAYER_1_SIZE)
  layer = tf.contrib.layers.fully_connected(
        inputs=layer, activation_fn=tf.nn.relu, num_outputs=LAYER_2_SIZE)
  logits = tf.contrib.layers.fully_connected(
        inputs=layer, activation_fn=tf.nn.relu, num_outputs=LAYER_3_SIZE)
  probabilities = tf.contrib.layers.fully_connected(
        inputs=logits, activation_fn=tf.sigmoid, num_outputs=1)
  return logits, probabilities

def run_train(train_input_data, labels):
  logits, probabilities = build_model(train_input_data, labels)
  loss = tf.losses.sigmoid_cross_entropy(labels, logits)
  optimizer = tf.train.AdamOptimizer(0.001)
  return optimizer.minimize(loss), loss, probabilities


def create_input():
  r = tf.range(_SIZE * _SIZE)
  random_inp = tf.stack([tf.random_shuffle(r) for _ in range(_BATCH_SIZE)])
  valid = tf.stack(is_valid(random_inp, _SIZE))
  valid = tf.reshape(valid, [_BATCH_SIZE, 1])
  return random_inp, valid


def main():
  with tf.Graph().as_default():

    # convert_indices = lambda x: to_index(x[0], x[1], _SIZE)
    # a = [tf.constant([0, 0]),tf.constant([1, 2]), tf.constant([3, 3]),
    # tf.constant([5, 4]), tf.constant([6, 6])]
    # moves_coords = tf.map_fn(convert_indices, tf.stack(a))
    # moves_coords = tf.reshape(moves_coords, [1, 5])
    # data_is_valid = is_valid(moves_coords, _SIZE)
    numbers, numbers_valid = create_input()
    with tf.control_dependencies([numbers, numbers_valid]):
      train_op, loss, _ = run_train(numbers, numbers_valid)
    with tf.Session() as sess:
      for i in range(_STEPS):
        sess.run(tf.global_variables_initializer())
        sess.run([numbers, numbers_valid, train_op, loss])
        if i % 100 == 0:
          print loss.eval()
        # print sess.run([data_is_valid, moves_coords]) 
    


if __name__ == "__main__":
  main()

