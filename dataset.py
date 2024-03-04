import functools
import json
import os
import tensorflow.compat.v1 as tf
from util import NodeType


def _parse(proto, meta):
    feature_lists = {k: tf.io.VarLenFeature(tf.string) for k in meta['field_names']}
    features = tf.io.parse_single_example(proto, feature_lists)
    out = {}
    for key, field in meta['features'].items():
        data = tf.io.decode_raw(features[key].values, getattr(tf, field['dtype']))
        data = tf.reshape(data, field['shape'])
        if field['type'] == 'static':
            data = tf.tile(data, [meta['trajectory_length'], 1, 1])
            out[key] = data
        elif field['type'] == 'static_varlen':
            length = tf.io.decode_raw(features['length_' + key].values, tf.int32)
            length = tf.reshape(length, [-1])
            data = tf.RaggedTensor.from_row_splits(data, length)
   
            data = tf.expand_dims(data, 0)
       
            data = tf.tile(data, [meta['trajectory_length'], 1, 1])
            out[key] = data
        elif field['type'] == 'dynamic':
            out[key] = data
        elif field['type'] != 'dynamic':
            raise ValueError('invalid data format')
    return out


def load_dataset(path, split):
    """Load dataset."""
    with open(os.path.join(path, 'meta.json'), 'r') as fp:
        meta = json.loads(fp.read())
    ds = tf.data.TFRecordDataset(os.path.join(path, split+'.tfrecord'))
    ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=1)
    ds = ds.prefetch(1)
    return ds


def add_targets(ds, fields, add_history):
    """Adds target and optionally history fields to dataframe."""
    def fn(trajectory):
        out = {}
        for key, val in trajectory.items():
            out[key] = val[1:-1]
            if key in fields:
                if add_history:
                    out['prev|' + key] = val[0:-2]
                out['target|' + key] = val[2:]

        return out

    return ds.map(fn, num_parallel_calls=1)


def split_and_preprocess(ds, noise_field, noise_scale, noise_gamma, seed):
    
    """Splits trajectories into frames, and adds training noise."""
    def add_noise(frame):
        noise = tf.random.normal(tf.shape(frame[noise_field]), stddev=noise_scale, dtype=tf.float32, seed=seed)
        # don't apply noise to boundary nodes
        mask = tf.equal(frame['node_type'], NodeType.NORMAL)[:, 0]
        noise = tf.where(mask, noise, tf.zeros_like(noise))
        frame[noise_field] += noise
        frame['target|'+noise_field] += (1.0 - noise_gamma) * noise
        return frame

    ds = ds.flat_map(tf.data.Dataset.from_tensor_slices)
    ds = ds.map(add_noise, num_parallel_calls=1)
    ds = ds.shuffle(2500, seed=seed, reshuffle_each_iteration=False)
    ds = ds.repeat(None)

    return ds.prefetch(10)

