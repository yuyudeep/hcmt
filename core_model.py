import tensorflow as tf
import sonnet as snt

from normalization import Normalizer


class FFN(snt.Module):
    def __init__(self, hidden_size, name=None):
        super(FFN, self).__init__(name=name)

        self.ffn1 = snt.Linear(hidden_size, with_bias=False, name='ffn_h1')
        self.ffn2 = snt.Linear(hidden_size, with_bias=False, name='ffn_h2')
        self.ffn3 = snt.Linear(hidden_size, with_bias=False, name='ffn_h3')

    def __call__(self, h):
        h = self.ffn1(h)
        h = tf.nn.relu(h)
        h = self.ffn2(h)
        h = tf.nn.relu(h)
        h = self.ffn3(h)
        return h


class MLP(snt.Module):
    def __init__(self, _num_layers, _latent_size, _output_size, layer_norm=True, name=None):
        super(MLP, self).__init__(name=name)
        
        widths = [_latent_size] * _num_layers + [_output_size]
        self.network = snt.nets.MLP(widths, activate_final=False)

        if layer_norm:
            self.network = snt.Sequential([self.network, snt.LayerNorm(axis=-1, create_scale=True, create_offset=True)])      

    def __call__(self, inputs):
        return self.network(inputs)


class EncoderNode(snt.Module):
    def __init__(self, num_layers, latent_size, name='encoder_node'):
        super(EncoderNode, self).__init__(name=name)
        
        self.node_latents = MLP(num_layers, latent_size, latent_size, name=f'{name}_node')
    
    
    def __call__(self, node_features):
        return self.node_latents(node_features)


class EncoderEdgeMesh(snt.Module):
    def __init__(self, num_layers, latent_size, feat_size, name='encoder_edge_mesh'):
        super(EncoderEdgeMesh, self).__init__(name=name)

        self._edge_mesh_normalizer = Normalizer(size=feat_size, name='edge_mesh_normalizer')
        self.edge_latents = MLP(num_layers, latent_size, latent_size, name=f'{name}_edge_mesh')


    def __call__(self, world_pos, mesh_pos, senders, receivers, is_training):

        relative_world_pos = tf.gather(world_pos, senders) - tf.gather(world_pos, receivers)
        relative_mesh_pos = tf.gather(mesh_pos, senders) - tf.gather(mesh_pos, receivers)
        
        edge_features = tf.concat([
            relative_world_pos,
            tf.norm(relative_world_pos, axis=-1, keepdims=True),
            relative_mesh_pos,
            tf.norm(relative_mesh_pos, axis=-1, keepdims=True)], axis=-1)

        return self.edge_latents(self._edge_mesh_normalizer(edge_features, is_training))


class EncoderEdgeCnt(snt.Module):
    def __init__(self, num_layers, latent_size, feat_size, name='encoder_edge_cnt'):
        super(EncoderEdgeCnt, self).__init__(name=name)

        self._edge_mesh_normalizer = Normalizer(size=feat_size, name='edge_cnt_normalizer')
        self.edge_latents = MLP(num_layers, latent_size, latent_size, name=f'{name}_edge_cnt')


    def __call__(self, world_pos, senders, receivers, is_training):

        relative_world_pos = tf.gather(world_pos, senders) - tf.gather(world_pos, receivers)
        edge_features = tf.concat([
            relative_world_pos,
            tf.norm(relative_world_pos, axis=-1, keepdims=True)
            ], axis=-1)

        return self.edge_latents(self._edge_mesh_normalizer(edge_features, is_training))


class MultiHeadAttentionCnt(snt.Module):
    def __init__(self, out_dim, num_heads, name='MultiHeadAttentionCnt'):
        super(MultiHeadAttentionCnt, self).__init__(name=name)
        
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.latent_size = int(out_dim * num_heads)


        self.Q = snt.Linear(out_dim * num_heads, with_bias=False)
        self.K = snt.Linear(out_dim * num_heads, with_bias=False)
        self.V = snt.Linear(out_dim * num_heads, with_bias=False)
        self.proj_mesh = snt.Linear(out_dim * num_heads, with_bias=False)
        self.G_mesh = snt.Linear(out_dim * num_heads, with_bias=False)


    def propagate_attention(self, num_nodes, Q_h, K_h, V_h, G_h, proj_e, senders, receivers):
        q = tf.gather(K_h, senders)
        k = tf.gather(Q_h, receivers)
        v = tf.gather(V_h, senders)
        
        scale_factor = tf.sqrt(tf.cast(self.out_dim, dtype=tf.float32))
        score = tf.multiply(q, k) / scale_factor

        score = tf.multiply(proj_e, score)
        score = tf.clip_by_value(score, -2, 2)

        attention_scores = tf.exp(score)
        v = tf.multiply(v, G_h)

        h_node = tf.math.unsorted_segment_sum(tf.multiply(v, attention_scores), receivers, num_nodes) 
        h_node = tf.reshape(h_node, [-1, self.latent_size])

        return h_node
    

    def __call__(self, V, E_mesh, E_cnt, m_senders, m_receivers, c_senders, c_receivers):
        
        q = self.Q(V)
        k = self.K(V)
        v = self.V(V)

        proj_mesh = self.proj_mesh(E_mesh)
        gmesh = self.G_mesh(E_mesh)
        proj_cnt = self.proj_mesh(E_cnt)
        gcnt = self.G_mesh(E_cnt)

        q = tf.reshape(q, [-1, self.num_heads, self.out_dim])
        k = tf.reshape(k, [-1, self.num_heads, self.out_dim])
        v = tf.reshape(v, [-1, self.num_heads, self.out_dim])
        gmesh = tf.reshape(gmesh, [-1, self.num_heads, self.out_dim])
        gcnt = tf.reshape(gcnt, [-1, self.num_heads, self.out_dim])

        proj_mesh = tf.reshape(proj_mesh, [-1, self.num_heads, self.out_dim])
        proj_cnt = tf.reshape(proj_cnt, [-1, self.num_heads, self.out_dim])


        num_nodes = tf.shape(V)[0]
        h_node1 = self.propagate_attention(num_nodes, q, k, v, gmesh, proj_mesh, m_senders, m_receivers)
        h_node2 = self.propagate_attention(num_nodes, q, k, v, gcnt, proj_cnt, c_senders, c_receivers)
        h_node = tf.concat([h_node1, h_node2], axis=1)
        return h_node


class GraphTransformerCnt(snt.Module):
    def __init__(self, out_dim, num_heads, name='GraphTransformerEdge'):
        super(GraphTransformerCnt, self).__init__(name=name)


        self.attention = MultiHeadAttentionCnt(out_dim//num_heads, num_heads)

        self.O_V = snt.Linear(out_dim, with_bias=False, name='O_node')

        self.ffn_v = FFN(out_dim)
        self.norm_node1 = snt.LayerNorm(-1, True, True)
        self.norm_node2 = snt.LayerNorm(-1, True, True)

        self.norm_edge1 = snt.LayerNorm(-1, True, True, name='norm_edge1')
        self.norm_edge1_cnt = snt.LayerNorm(-1, True, True, name='norm_edge1')


    def __call__(self, V, E_mesh, E_cnt, m_senders, m_receivers, c_senders, c_receivers):
        
        V1 = self.norm_node1(V)
        E1 = self.norm_edge1(E_mesh)
        E1_cnt = self.norm_edge1_cnt(E_cnt)
        
        V2 = self.attention(V1, E1, E1_cnt, m_senders, m_receivers, c_senders, c_receivers)

        V3 = V + self.O_V(V2)

        V4 = self.norm_node2(V3)
        V5 = self.ffn_v(V4)
        V = V3 + V5
        
        return V


class MultiHeadAttentionMesh(snt.Module):
    def __init__(self, out_dim, num_heads, name='MultiHeadAttentionMesh'):
        super(MultiHeadAttentionMesh, self).__init__(name=name)
        
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.latent_size = int(out_dim * num_heads)


        self.Q = snt.Linear(out_dim * num_heads, with_bias=False)
        self.K = snt.Linear(out_dim * num_heads, with_bias=False)
        self.V = snt.Linear(out_dim * num_heads, with_bias=False)
        self.proj_mesh = snt.Linear(out_dim * num_heads, with_bias=False)
        self.G_mesh = snt.Linear(out_dim * num_heads, with_bias=False)


    def single_propagate_attention(self, num_nodes, Q_h, K_h, V_h, G_h, proj, senders, receivers):
        q = tf.gather(K_h, senders)
        k = tf.gather(Q_h, receivers)
        v = tf.gather(V_h, senders)
        
        scale_factor = tf.sqrt(tf.cast(self.out_dim, dtype=tf.float32))
        score = tf.multiply(q, k) / scale_factor

        score = tf.multiply(proj, score)
        score = tf.clip_by_value(score, -2, 2)

        attention_scores = tf.exp(score)

        v = tf.multiply(v, G_h)

        h_node = tf.math.unsorted_segment_sum(tf.multiply(v, attention_scores), receivers, num_nodes) 

        h_node = tf.reshape(h_node, [-1, self.latent_size])

        return h_node
    

    def __call__(self, h, e, senders, receivers):
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)

        proj = self.proj_mesh(e)
        G_h = self.G_mesh(e)

        Q_h = tf.reshape(Q_h, [-1, self.num_heads, self.out_dim])
        K_h = tf.reshape(K_h, [-1, self.num_heads, self.out_dim])
        V_h = tf.reshape(V_h, [-1, self.num_heads, self.out_dim])
        G_h = tf.reshape(G_h, [-1, self.num_heads, self.out_dim])

        proj = tf.reshape(proj, [-1, self.num_heads, self.out_dim])


        num_nodes = tf.shape(h)[0]
        h_node = self.single_propagate_attention(num_nodes, Q_h, K_h, V_h, G_h, proj, senders, receivers)
        
        return h_node


class GraphTransformerEdge(snt.Module):
    def __init__(self, out_dim, num_heads, name='GraphTransformerEdge'):
        super(GraphTransformerEdge, self).__init__(name=name)

        self.attention = MultiHeadAttentionMesh(out_dim//num_heads, num_heads)

        self.O_V = snt.Linear(out_dim, with_bias=False, name='O_node')

        self.ffn_v = FFN(out_dim)
        self.norm_node1 = snt.LayerNorm(-1, True, True)
        self.norm_node2 = snt.LayerNorm(-1, True, True)

        self.norm_edge1 = snt.LayerNorm(-1, True, True, name='norm_edge1')

    
    def __call__(self, V, E, senders, receivers):
        
        V1 = self.norm_node1(V)
        E1 = self.norm_edge1(E)

        V2 = self.attention(V1, E1, senders, receivers)

        V3 = V + self.O_V(V2)
        
        V4 = self.norm_node2(V3)
        V5 = self.ffn_v(V4)
        V = V3 + V5
        return V


class HCMT(snt.Module):
    def __init__(self, name='HierarchyMeshTransformer'):
        super(HCMT, self).__init__(name=name)

        """ Parameters """
        num_layers = 2
        latent_size = 128
        output_size = 3
        num_heads = 4

        self.encoder_node = EncoderNode(num_layers, latent_size)
        self.encoder_edge_mesh = EncoderEdgeMesh(num_layers, latent_size, 6)
        self.encoder_edge_cnt  = EncoderEdgeCnt(num_layers, latent_size, 3)
        self.decoder = MLP(num_layers, latent_size, output_size, layer_norm=False, name='decoder')

        self.l_n = 6

        self.downs = [GraphTransformerEdge(latent_size, num_heads) for _ in range(self.l_n)]
        self.bottom = GraphTransformerEdge(latent_size, num_heads)
        self.ups = [GraphTransformerEdge(latent_size, num_heads) for _ in range(self.l_n)]

        self.firsts = [GraphTransformerCnt(latent_size, num_heads) for _ in range(2)]
        print('hierarchy level', self.l_n)


    def __call__(self, node_features, inputs, edges, is_training):

        V = self.encoder_node(node_features)
                
        adj_ms_sen = []
        adj_ms_rev = []
        indices_list = []
        down_outs_v = []
        down_outs_e = []

        world_pos = inputs['world_pos']
        mesh_pos = inputs['mesh_pos']

        m_senders = edges['m_senders']
        c_senders = edges['c_senders']
        m_receivers = edges['m_receivers']
        c_receivers = edges['c_receivers']

        E_mesh = self.encoder_edge_mesh(world_pos, mesh_pos, m_senders, m_receivers, is_training)
        E_cnt = self.encoder_edge_cnt(world_pos, c_senders, c_receivers, is_training)
        
        for net in self.firsts:
            V = net(V, E_mesh, E_cnt, m_senders, m_receivers, c_senders, c_receivers)

        for i in range(self.l_n):
            # Pool
            idx = inputs['m_ids'][i]
            V = tf.gather(V, idx)

            senders = inputs['m_gs_s'][i]
            receivers = inputs['m_gs_r'][i]
            world_pos = tf.gather(world_pos, idx)
            mesh_pos = tf.gather(mesh_pos, idx)
            
            E = self.encoder_edge_mesh(world_pos, mesh_pos, senders, receivers, is_training)
            V = self.downs[i](V, E, senders, receivers)

            down_outs_v.append(V)
            down_outs_e.append(E)

            adj_ms_sen.append(senders)
            adj_ms_rev.append(receivers)
            indices_list.append(idx)

        # Bottom
        idx = inputs['m_ids'][i + 1]
        V = tf.gather(V, idx)
        senders = inputs['m_gs_s'][i + 1]
        receivers = inputs['m_gs_r'][i + 1]
        world_pos = tf.gather(world_pos, idx)
        mesh_pos = tf.gather(mesh_pos, idx)
        E = self.encoder_edge_mesh(world_pos, mesh_pos, senders, receivers, is_training)
        V = self.bottom(V, E, senders, receivers)
        indices_list.append(idx)

        for i in range(self.l_n):
            # UnPool
            up_idx = self.l_n - i - 1
            senders, receivers, idx = adj_ms_sen[up_idx], adj_ms_rev[up_idx], indices_list[up_idx + 1]
            pre_v = down_outs_v[up_idx]
            E = down_outs_e[up_idx]

            V = tf.tensor_scatter_nd_update(pre_v, tf.expand_dims(idx, -1), V)

            V = self.ups[i](V, E, senders, receivers)

        return self.decoder(V)
