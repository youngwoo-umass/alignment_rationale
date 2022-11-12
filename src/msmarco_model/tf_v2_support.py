import tensorflow as tf


placeholder = tf.compat.v1.placeholder
def disable_eager_execution():
    tf.compat.v1.disable_eager_execution()


variable_scope = tf.compat.v1.variable_scope

tf_record_enum = tf.data.TFRecordDataset

tf1 = tf.compat.v1