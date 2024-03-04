import sonnet as snt
import tensorflow as tf
import collections

from util import NodeType, triangles_to_edges, get_mask_impact_normal
from normalization import Normalizer


EdgeSet = collections.namedtuple('EdgeSet', ['name', 'features', 'senders', 'receivers'])
MultiGraph = collections.namedtuple('Graph', ['node_features', 'edge_sets'])


def pairwise_dist(A, B):
    na = tf.reduce_sum(tf.square(A), 1)
    nb = tf.reduce_sum(tf.square(B), 1)
    
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])

    D = tf.sqrt(tf.maximum(na - 2*tf.matmul(A, B, False, True) + nb, 0.0))
    return D


@tf.function(input_signature=(
        tf.TensorSpec(shape=[None, 2], dtype=tf.float32, name="world_pos"),
        tf.TensorSpec(shape=[None, 1], dtype=tf.int32, name="node_type"),
        tf.TensorSpec(shape=[], dtype=tf.float64, name='radius')
        ))
def contact_node_2D(pos, node_type, radius):
    """
    Define the contact edges
    Input:
    - pos: A tensor with shape [#nodes, 2], meaning world positions at t.
    - node_type: A tensor with shape [#nodes, 1], meaning the node_type at t.
    - radius: A tensor with shape []
    Output:
    - contact edges
    """

    table = tf.lookup.experimental.DenseHashTable(
        key_dtype=tf.int32,
        value_dtype=tf.int32,
        default_value=-1,
        empty_key=-2,
        deleted_key=-3
        )
    pos = tf.cast(pos, tf.float64)
    numbers_node = tf.range(tf.shape(pos)[0])

    mask_node_sen = tf.equal(node_type[:, 0], NodeType.OBSTACLE)
    mask_node_rev = tf.equal(node_type[:, 0], NodeType.NORMAL)

    pos_node_sen = pos[mask_node_sen]
    pos_node_sen_idx = numbers_node[mask_node_sen]
    pos_node_rev = pos[mask_node_rev]
    pos_node_rev_idx = numbers_node[mask_node_rev]
    
    dist = pairwise_dist(pos_node_sen, pos_node_rev)
    sp = tf.sparse.from_dense(dist)

    mask = tf.less(sp.values, radius)
    pair = tf.cast(sp.indices[mask], tf.int32)
    senders_node, receivers_node = tf.unstack(pair, axis=1)

    keys = tf.range(tf.shape(pos_node_sen)[0])
    table.insert(keys, pos_node_sen_idx)
    senders = table.lookup(senders_node)
    table.remove(keys)

    keys = tf.range(tf.shape(pos_node_rev)[0])
    table.insert(keys, pos_node_rev_idx)
    receivers = table.lookup(receivers_node)
    table.remove(keys)
    del table
    return (tf.concat([senders, receivers], axis=0), tf.concat([receivers, senders], axis=0))


class ImpactModel(snt.Module):
    def __init__(self, learned_model, name='Model'):
        super(ImpactModel, self).__init__(name=name)
        self._learned_model = learned_model

        # Normalizer
        self._output_normalizer = Normalizer(size=2, name='output_normalizer')
        self._output_stress_normalizer = Normalizer(size=1, name='output_normalizer')
        self._node_normalizer = Normalizer(size=4 + NodeType.SIZE, name='node_normalizer')


    def _build_graph(self, inputs, is_training):
        velocity = (inputs['world_pos'] - inputs['prev|world_pos'])
        node_type = tf.one_hot(inputs['node_type'][:, 0], NodeType.SIZE)

        node_features = tf.concat([velocity, node_type, inputs['density'], inputs['modulus']], axis=-1)

        senders, receivers = triangles_to_edges(inputs['cells'])

        radius = tf.cast(0.4, tf.float64)
        senders_n, receivers_n = contact_node_2D(inputs['world_pos'], inputs['node_type'], radius)

        edges = {}
        edges['m_senders'] = senders
        edges['m_receivers'] = receivers
        edges['c_senders'] = senders_n
        edges['c_receivers'] = receivers_n
 
        return self._node_normalizer(node_features, is_training), inputs, edges
        

    def __call__(self, inputs):
        node_features, inputs, edges = self._build_graph(inputs, is_training=False)
        per_node_network_output = self._learned_model(node_features, inputs, edges, is_training=False)

        return self._update(inputs, per_node_network_output)
    

    def loss(self, inputs):
        node_features, inputs, edges = self._build_graph(inputs, is_training=True)
        network_output = self._learned_model(node_features, inputs, edges, is_training=True)

        target_velocity = inputs['target|world_pos'] - inputs['world_pos']
        target_normalized = self._output_normalizer(target_velocity)

        target_stress = inputs['target|stress']
        target_stress_normalized = self._output_stress_normalizer(target_stress)

        loss_mask = get_mask_impact_normal(inputs)
        error_vel = (target_normalized - network_output[:, :2]) ** 2
        error_vel = tf.where(loss_mask, error_vel, tf.zeros_like(error_vel))
        error_vel = tf.reduce_sum(error_vel, axis=-1)
        loss_vel = tf.reduce_mean(error_vel)

        error_pre = (target_stress_normalized - network_output[:, 2:3]) ** 2
        error_pre = tf.reduce_sum(error_pre, axis=-1)
        loss_stress = tf.reduce_mean(error_pre)
        
        return loss_vel + loss_stress
    

    def _update(self, inputs, per_node_network_output):
        pred_velocity = self._output_normalizer.inverse(per_node_network_output[:, :2])
        pred_stress = self._output_stress_normalizer.inverse(per_node_network_output[:, 2:3])
        position = inputs['world_pos'] + pred_velocity
        return position, pred_stress


def evaluate_impact(model, inputs):
    def _rollout(model, initial_state, num_steps):
        mask = get_mask_impact_normal(initial_state)
        def step_fn(step, prev_pos, cur_pos, cur_stress, trajectory, trajectory_stress):
            prediction, pred_stress = model({**initial_state, 'prev|world_pos': prev_pos, 'world_pos': cur_pos})
            next_pos = tf.where(mask, prediction, cur_pos)
            trajectory = trajectory.write(step, cur_pos)
            trajectory_stress = trajectory_stress.write(step, cur_stress)
            return step+1, cur_pos, next_pos, pred_stress, trajectory, trajectory_stress
        
        _, _, _, _, output, output_stress = tf.while_loop(
        cond=lambda step, prev, cur, cur_stress, traj_vel, traj_str: tf.less(step, num_steps), 
        body=step_fn,
        loop_vars=(0, initial_state['prev|world_pos'], initial_state['world_pos'], initial_state['stress'], 
                   tf.TensorArray(tf.float32, num_steps), tf.TensorArray(tf.float32, num_steps)),
        parallel_iterations=1)
        return output.stack(), output_stress.stack()

    initial_state = {k: v[0] for k, v in inputs.items()}
    num_steps = inputs['cells'].shape[0]
    pred_pos, pred_stress = _rollout(model, initial_state, num_steps)
   

    error_pos_rmse = tf.sqrt(tf.reduce_mean(tf.reduce_sum((pred_pos - inputs['world_pos'])**2, axis=-1), -1))
    error_stress_rmse = tf.sqrt(tf.reduce_mean(tf.reduce_sum((pred_stress - inputs['stress'])**2, axis=-1), -1))

    scalars = {'p%d' % horizon: tf.reduce_mean(error_pos_rmse[1:horizon+1]).numpy() * 1E3 for horizon in [1, 50, error_pos_rmse.shape[0]]}
    scalars_stress = {'s%d' % horizon: tf.reduce_mean(error_stress_rmse[1:horizon+1]).numpy() * 1E3 for horizon in [1, 50, error_stress_rmse.shape[0]]}
    scalars.update(scalars_stress)

    traj_ops = {
        'faces': inputs['cells'],
        'mesh_pos': inputs['mesh_pos'],
        'gt_pos': inputs['world_pos'],
        'pred_pos': pred_pos,
        'gt_stress': inputs['stress'],
        'pred_stress': pred_stress,
        'node_type': inputs['node_type']
    }

    return scalars, traj_ops
