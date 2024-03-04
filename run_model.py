import tensorflow as tf
import numpy as np
import os
import dataset
import random
import pickle

from pathlib import Path
from util import get_check_point_num
from termcolor import colored
from core_model import HCMT
from model import ImpactModel, evaluate_impact

from absl import app
from absl import flags
from absl import logging

print(colored(f'tensorflow version : {tf.__version__}', 'red'))
print(colored(f'GPUs Available : {len(tf.config.experimental.list_physical_devices("GPU"))}', 'red'))

gpus = tf.config.list_physical_devices('GPU')

FLAGS = flags.FLAGS
flags.DEFINE_enum('mode', 'train', ['train', 'eval'], 'Train model, or run evaluation.')
flags.DEFINE_string('dataset_dir', "data/impact", 'Directory to load dataset from.')
flags.DEFINE_string('checkpoint_dir', 'workspace/run/check', 'Directory to save checkpoint')
flags.DEFINE_string('rollout_dir', 'workspace/run/rollout', 'Pickle file to save eval trajectories')
flags.DEFINE_string('logging_dir', 'workspace/run/log', 'log directory')
flags.DEFINE_integer('num_training_steps', 2000000, 'No. of training steps')
flags.DEFINE_integer('num_rollouts', 200, 'No. of rollouts')
flags.DEFINE_integer('seed', 10, 'No. of random seed')

for i in range(len(gpus)):
	tf.config.experimental.set_memory_growth(gpus[i], True)


def learner(model):

    @tf.function(input_signature=[{
        "cells": tf.TensorSpec(shape=[None, 3], dtype=tf.int32, name="cells"),
        "stress": tf.TensorSpec(shape=[None, 1], dtype=tf.float32, name="eqv_stress"),
        "mesh_pos": tf.TensorSpec(shape=[None, 2], dtype=tf.float32, name="mesh_pos"),
        "node_type": tf.TensorSpec(shape=[None, 1], dtype=tf.int32, name="node_type"),
        "prev|stress": tf.TensorSpec(shape=[None, 1], dtype=tf.float32, name="prev|eqv_stress"),
        "prev|world_pos": tf.TensorSpec(shape=[None, 2], dtype=tf.float32, name="prev|world_pos"),
        "target|stress": tf.TensorSpec(shape=[None, 1], dtype=tf.float32, name="target|eqv_stress"),
        "target|world_pos": tf.TensorSpec(shape=[None, 2], dtype=tf.float32, name="target|world_pos"),
        "world_pos": tf.TensorSpec(shape=[None, 2], dtype=tf.float32, name="world_pos"),
        "density": tf.TensorSpec(shape=[None, 1], dtype=tf.float32, name="density"),
        "modulus": tf.TensorSpec(shape=[None, 1], dtype=tf.float32, name="modulus"),
        "lap_pe": tf.TensorSpec(shape=[None, 8], dtype=tf.float32, name="lap_pe"),
        "m_gs_s": tf.RaggedTensorSpec(shape=[7, None], dtype=tf.int32, ragged_rank=1, row_splits_dtype=tf.int32),
        "m_gs_r": tf.RaggedTensorSpec(shape=[7, None], dtype=tf.int32, ragged_rank=1, row_splits_dtype=tf.int32),
        "m_ids": tf.RaggedTensorSpec(shape=[7, None], dtype=tf.int32, ragged_rank=1, row_splits_dtype=tf.int32),
    }])
    def train_step(inputs):
        with tf.GradientTape() as tape:
            loss = model.loss(inputs)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss
            
    ds = dataset.load_dataset(FLAGS.dataset_dir, 'train')

    ds = dataset.add_targets(ds, ['world_pos', 'stress'], add_history=True)
    ds = dataset.split_and_preprocess(
        ds, 
        noise_field='world_pos',
        noise_scale=0.003,
        noise_gamma=1,
        seed=FLAGS.seed
        )

    ds = tf.compat.v1.data.make_one_shot_iterator(ds)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    ckpt = tf.train.Checkpoint(step=global_step, net=model)
    manager = tf.train.CheckpointManager(checkpoint=ckpt, directory=FLAGS.checkpoint_dir, max_to_keep=50)
    ckpt.restore(manager.latest_checkpoint)
    
    lr_schedule = tf.compat.v1.train.exponential_decay(learning_rate=1e-4,
                                global_step=global_step,
                                decay_steps=int(1000000),
                                decay_rate=0.1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


    losses = 0

    """ Training """
    counter = 0
    epoch_steps = 100000
    for step in range(int(global_step), FLAGS.num_training_steps + 1, 1):
        inputs = ds.get_next()

        if step < 1000:
            model._build_graph(inputs, True)
        else:
            loss = train_step(inputs)
            losses += loss
            counter += 1

            if counter != 1 and step % epoch_steps == 0:
                manager.save(checkpoint_number=int(global_step))
                print(f'{step} {losses/counter}')
            
            if counter != 1 and step % int(epoch_steps/200) == 0:
                print(f'{step} {losses/counter:.9f}')

        global_step.assign_add(1)

    manager.save(checkpoint_number=int(global_step))
    with open(os.path.join(FLAGS.logging_dir, 'train_epoch_RMSE.txt'), 'a') as file:
        file.write(f'{step} {losses/counter}\n')
    

def evaluator(model):

    ds = dataset.load_dataset(FLAGS.dataset_dir, 'test')
    ds = dataset.add_targets(ds, ['world_pos', 'stress'], add_history=True)

    ds = tf.compat.v1.data.make_one_shot_iterator(ds)

    trajectories = []
    scalars = []

    global_step = tf.Variable(0, name='global_step', trainable=False)
    ckpt = tf.train.Checkpoint(step=global_step, net=model)
    manager = tf.train.CheckpointManager(checkpoint=ckpt, directory=FLAGS.checkpoint_dir, max_to_keep=None)
    ckpt.restore(manager.latest_checkpoint)
    checkpoint_num = get_check_point_num(os.path.join(FLAGS.checkpoint_dir, 'checkpoint'))

    print(colored(checkpoint_num, 'red'))
    counter = 0
    for traj_idx in range(FLAGS.num_rollouts):
        inputs = ds.get_next()
        scalar_data, traj_data = evaluate_impact(model, inputs)
        trajectories.append(traj_data)

        scalars.append(scalar_data)
        print(traj_idx, scalar_data)
        counter += 1
        del traj_data
        del inputs
    
    with open(os.path.join(FLAGS.logging_dir, 'test_RMSE.txt'), 'a') as file:
        txt = ''
        for key in scalars[0]:
            print('%s: %g', key, np.mean([x[key] for x in scalars]))
            txt += f' {key} {np.mean([x[key] for x in scalars])}'
        file.write(f'{checkpoint_num} {txt}\n')
    
    with open(os.path.join(FLAGS.rollout_dir, f'{checkpoint_num}.pkl'), 'wb') as fp:
        pickle.dump(trajectories, fp)


def main(argv):
    del argv

    tf.compat.v1.enable_resource_variables()
    tf.config.run_functions_eagerly(False)
    
    """ Create base directory """
    Path(FLAGS.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(FLAGS.rollout_dir).mkdir(parents=True, exist_ok=True)
    Path(FLAGS.logging_dir).mkdir(parents=True, exist_ok=True)

    """ Fix seed """
    tf.keras.utils.set_random_seed(FLAGS.seed)
    tf.config.experimental.enable_op_determinism()
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    tf.random.set_seed(FLAGS.seed)
    tf.compat.v1.set_random_seed(FLAGS.seed)
    
    model = ImpactModel(HCMT())

    if FLAGS.mode == 'train':
        learner(model)

    if FLAGS.mode == 'eval':
        evaluator(model)


if __name__ == '__main__':
    app.run(main)
