from collections import OrderedDict
import time, os, talib
import multiprocessing as mp
curdir = os.path.expanduser("~")
import numpy as np
np.set_printoptions(precision=8, suppress=True, linewidth=400, threshold=100)
# np.random.seed(0)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 0,1,2,3
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit' # lets XLA work on CPU
import tensorflow as tf
# tf.keras.backend.set_floatx('float64')
# tf.config.run_functions_eagerly(True)
# tf.config.optimizer.set_jit("autoclustering") # enable XLA
# tf.config.experimental.enable_mlir_graph_optimization()
# tf.random.set_seed(0)
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import model_util as util
import gym

# CUDA 11.2.2_461.33, CUDNN 8.1.1.33, tf-nightly-gpu==2.7.0.dev20210720, tfp_nightly==0.14.0.dev20210630
physical_devices_gpu = tf.config.list_physical_devices('GPU')
for i in range(len(physical_devices_gpu)): tf.config.experimental.set_memory_growth(physical_devices_gpu[i], True)

# TODO try out MuZero-ish architecture
# TODO add Perciever, maybe ReZero

# TODO add GenNet and DisNet for GAN type boost
# TODO put actor in seperate process so can run async
# TODO add ZMQ and latent pooling

# TODO how to incorporate ARS random policy search?
# TODO try out the 'lottery ticket hypothosis' pruning during training
# TODO use numba to make things faster on CPU


class RepNet(tf.keras.Model):
    def __init__(self, name, spec_in, latent_spec, latent_dist, latent_size, net_blocks, net_attn, net_lstm, num_heads=2, memory_size=1):
        super(RepNet, self).__init__()
        inp, mid, evo = latent_size*4, latent_size*4, int(latent_size/2)
        self.net_blocks, self.net_attn, self.net_lstm = net_blocks, net_attn, net_lstm
        self.layer_flatten = tf.keras.layers.Flatten()

        # TODO how to loop through different embed layer structures? so can combine RepNet and TransNet
        # TODO possibly use Perciever https://github.com/Rishit-dagli/Perceiver
        # self.net_inputs = ['obs']*len(spec_in)+['rewards','dones']
        self.net_ins, self.layer_dense_in, self.layer_dense_in_lat = len(spec_in), [], []
        # self.net_ins, self.layer_attn_in = len(spec_in), []
        # self.net_ins, self.layer_attn_in, self.layer_attn_in2 = len(spec_in), [], []
        for i in range(self.net_ins):
            self.layer_dense_in.append(tf.keras.layers.Dense(inp, activation=util.EvoNormS0(evo), use_bias=False, name='dense_in_{:02d}'.format(i)))
            self.layer_dense_in_lat.append(tf.keras.layers.Dense(latent_size, name='dense_in_lat_{:02d}'.format(i)))
            # self.layer_attn_in.append(util.CrossAttention(num_heads=num_heads, latent_size=latent_size, init=True, name='attn_in_{:02d}'.format(i)))
            # self.layer_attn_in2.append(util.CrossAttention(num_heads=num_heads, latent_size=latent_size, init=False, name='attn_in2_{:02d}'.format(i)))

        self.layer_attn, self.layer_lstm, self.layer_dense, self.layer_dense_lat = [], [], [], []
        for i in range(net_blocks):
            if self.net_attn: self.layer_attn.append(util.MultiHeadAttention(num_heads=num_heads, latent_size=latent_size, memory_size=memory_size, name='attn_{:02d}'.format(i)))
            if self.net_lstm: self.layer_lstm.append(tf.keras.layers.LSTM(mid, activation=util.EvoNormS0(evo), use_bias=False, return_sequences=True, stateful=True, name='lstm_{:02d}'.format(i)))
            else: self.layer_dense.append(tf.keras.layers.Dense(mid, activation=util.EvoNormS0(evo), use_bias=False, name='dense_{:02d}'.format(i)))
            self.layer_dense_lat.append(tf.keras.layers.Dense(latent_size, name='dense_lat_{:02d}'.format(i)))

        if latent_dist == 0: params_size, self.dist = util.Deterministic.params_size(latent_spec['event_shape']), util.Deterministic(latent_spec['event_shape'])
        if latent_dist == 1: params_size, self.dist = util.CategoricalRP.params_size(latent_spec['event_shape']), util.CategoricalRP(latent_spec['event_shape'])
        if latent_dist == 2: params_size, self.dist = util.MixtureLogistic.params_size(latent_spec['num_components'], latent_spec['event_shape']), util.MixtureLogistic(latent_spec['num_components'], latent_spec['event_shape'])
        self.layer_dense_logits_out = tf.keras.layers.Dense(params_size, name='dense_logits_out')

        self.call = tf.function(self.call, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
        self.net_arch = "{}[inD{}-{:02d}{}{}D{}{}-lat{}-cmp{}]".format(name, inp, net_blocks, ('AT+' if self.net_attn else ''), ('LS+' if self.net_lstm else ''), mid, ('-hds'+str(num_heads) if self.net_attn else ''), latent_size, latent_spec['num_components'])

    def reset_states(self):
        for layer in self.layer_attn: layer.reset_states()
        for layer in self.layer_lstm: layer.reset_states()
        # for layer in self.layer_attn_in: layer.reset_states()
    def call(self, inputs, training=None):
        out_accu = [None]*self.net_ins
        for i in range(self.net_ins):
            data = tf.cast(inputs['obs'][i], self.compute_dtype)
            out = self.layer_flatten(data)
            out = self.layer_dense_in[i](out)
            out = self.layer_dense_in_lat[i](out)
            # out = self.layer_attn_in[i](None, data)
            # out = self.layer_attn_in2[i](out, data)
            out_accu[i] = out
        # out = tf.math.accumulate_n(out_accu)
        out = tf.math.add_n(out_accu)
        
        for i in range(self.net_blocks):
            out = tf.expand_dims(out, axis=0)
            if self.net_attn: out = self.layer_attn[i](out, training=training)
            if self.net_lstm: out = self.layer_lstm[i](out, training=training)
            out = tf.squeeze(out, axis=0)
            if not self.net_lstm: out = self.layer_dense[i](out)
            out = self.layer_dense_lat[i](out)
        out = self.layer_dense_logits_out(out)

        isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(out), tf.math.is_inf(out)))
        if isinfnan > 0: tf.print('rep net out:', out)
        return out


# transition dynamics within latent space
class TransNet(tf.keras.Model):
    def __init__(self, name, spec_in, latent_spec, latent_dist, latent_size, net_blocks, net_attn, net_lstm, num_heads=2, memory_size=1): # spec_in=[] for no action conditioning
        super(TransNet, self).__init__()
        inp, mid, evo = latent_size*4, latent_size*4, int(latent_size/2)
        self.net_blocks, self.net_attn, self.net_lstm = net_blocks, net_attn, net_lstm
        self.layer_flatten = tf.keras.layers.Flatten()

        # self.net_inputs = ['actions']*len(spec_in)+['obs']
        self.net_ins, self.layer_dense_in, self.layer_dense_in_lat = len(spec_in), [], [] # action conditioning/embedding
        for i in range(self.net_ins):
            self.layer_dense_in.append(tf.keras.layers.Dense(inp, activation=util.EvoNormS0(evo), use_bias=False, name='dense_in_{:02d}'.format(i)))
            self.layer_dense_in_lat.append(tf.keras.layers.Dense(latent_size, name='dense_in_lat_{:02d}'.format(i)))

        self.layer_obs_dense_in = tf.keras.layers.Dense(inp, activation=util.EvoNormS0(evo), use_bias=False, name='obs_dense_in')
        self.layer_obs_dense_in_lat = tf.keras.layers.Dense(latent_size, name='obs_dense_in_lat')

        self.layer_attn, self.layer_lstm, self.layer_dense, self.layer_dense_lat = [], [], [], []
        for i in range(net_blocks):
            if self.net_attn: self.layer_attn.append(util.MultiHeadAttention(num_heads=num_heads, latent_size=latent_size, memory_size=memory_size, name='attn_{:02d}'.format(i)))
            if self.net_lstm: self.layer_lstm.append(tf.keras.layers.LSTM(mid, activation=util.EvoNormS0(evo), use_bias=False, return_sequences=True, stateful=True, name='lstm_{:02d}'.format(i)))
            else: self.layer_dense.append(tf.keras.layers.Dense(mid, activation=util.EvoNormS0(evo), use_bias=False, name='dense_{:02d}'.format(i)))
            self.layer_dense_lat.append(tf.keras.layers.Dense(latent_size, name='dense_lat_{:02d}'.format(i)))
        
        if latent_dist == 0: params_size, self.dist = util.Deterministic.params_size(latent_spec['event_shape']), util.Deterministic(latent_spec['event_shape'])
        if latent_dist == 1: params_size, self.dist = util.CategoricalRP.params_size(latent_spec['event_shape']), util.CategoricalRP(latent_spec['event_shape'])
        if latent_dist == 2: params_size, self.dist = util.MixtureLogistic.params_size(latent_spec['num_components'], latent_spec['event_shape']), util.MixtureLogistic(latent_spec['num_components'], latent_spec['event_shape'])
        self.layer_dense_logits_out = tf.keras.layers.Dense(params_size, name='dense_logits_out')

        self.call = tf.function(self.call, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
        self.net_arch = "{}[inD{}-{:02d}{}{}D{}{}-lat{}-cmp{}]".format(name, inp, net_blocks, ('AT+' if self.net_attn else ''), ('LS+' if self.net_lstm else ''), mid, ('-hds'+str(num_heads) if self.net_attn else ''), latent_size, latent_spec['num_components'])

    def reset_states(self):
        for layer in self.layer_attn: layer.reset_states()
        for layer in self.layer_lstm: layer.reset_states()
    def call(self, inputs, training=None):
        out_accu = [None]*(self.net_ins+1)
        for i in range(self.net_ins):
            out = tf.cast(inputs['actions'][i], self.compute_dtype)
            out = self.layer_flatten(out)
            out = self.layer_dense_in[i](out)
            out = self.layer_dense_in_lat[i](out)
            out_accu[i] = out

        out = tf.cast(inputs['obs'], self.compute_dtype)
        out = self.layer_flatten(out)
        out = self.layer_obs_dense_in(out)
        out = self.layer_obs_dense_in_lat(out)
        out_accu[-1] = out

        # out = tf.math.accumulate_n(out_accu)
        out = tf.math.add_n(out_accu)
        
        for i in range(self.net_blocks):
            out = tf.expand_dims(out, axis=0)
            if self.net_attn: out = self.layer_attn[i](out, training=training)
            if self.net_lstm: out = self.layer_lstm[i](out, training=training)
            out = tf.squeeze(out, axis=0)
            if not self.net_lstm: out = self.layer_dense[i](out)
            out = self.layer_dense_lat[i](out)
        out = self.layer_dense_logits_out(out)

        isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(out), tf.math.is_inf(out)))
        if isinfnan > 0: tf.print('trans net out:', out)
        return out


class RewardNet(tf.keras.Model):
    def __init__(self, name):
        super(RewardNet, self).__init__()
        num_components, event_shape = 16, (1,); params_size, self.dist = util.MixtureLogistic.params_size(num_components, event_shape), util.MixtureLogistic(num_components, event_shape)
        inp, evo = 256, 32; self.net_arch = "{}[inD{}-cmp{}]".format(name, inp, num_components)
        self.layer_flatten = tf.keras.layers.Flatten()
        self.layer_dense_in = tf.keras.layers.Dense(inp, activation=util.EvoNormS0(evo), use_bias=False, name='dense_in')
        self.layer_dense_logits_out = tf.keras.layers.Dense(params_size, name='dense_logits_out')
        # self.layer_dense_out = tf.keras.layers.Dense(1, name='dense_out')
        self.call = tf.function(self.call, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
    def call(self, inputs, training=None):
        out = self.layer_flatten(inputs['obs'])
        out = self.layer_dense_in(out)
        out = self.layer_dense_logits_out(out)
        return out

class DoneNet(tf.keras.Model):
    def __init__(self, name):
        super(DoneNet, self).__init__()
        num_components, event_shape = 2, (1,); params_size, self.dist = util.Categorical.params_size(num_components, event_shape), util.Categorical(num_components, event_shape)
        # params_size, self.dist = 1, tfp.layers.DistributionLambda(lambda input: tfp.distributions.Bernoulli(logits=input, dtype=tf.bool))
        inp, evo = 256, 32; self.net_arch = "{}[inD{}-cmp{}]".format(name, inp, num_components)
        self.layer_flatten = tf.keras.layers.Flatten()
        self.layer_dense_in = tf.keras.layers.Dense(inp, activation=util.EvoNormS0(evo), use_bias=False, name='dense_in')
        self.layer_dense_logits_out = tf.keras.layers.Dense(params_size, name='dense_logits_out')
        # self.layer_dense_out = tf.keras.layers.Dense(1, name='dense_out')
        self.call = tf.function(self.call, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
    def call(self, inputs, training=None):
        out = self.layer_flatten(inputs['obs'])
        out = self.layer_dense_in(out)
        out = self.layer_dense_logits_out(out)
        return out


class GenNet(tf.keras.Model):
    def __init__(self, name, spec_out, force_cont, latent_size, net_blocks=0, net_attn=False, net_lstm=False, num_heads=2, memory_size=1, force_det_out=False):
        super(GenNet, self).__init__()
        inp, mid, evo, mixture_size = latent_size*4, latent_size*4, int(latent_size/2), int(latent_size/2)
        self.net_blocks, self.net_attn, self.net_lstm = net_blocks, net_attn, net_lstm
        self.layer_flatten = tf.keras.layers.Flatten()

        self.layer_dense_in = tf.keras.layers.Dense(inp, activation=util.EvoNormS0(evo), use_bias=False, name='dense_in')
        self.layer_dense_in_lat = tf.keras.layers.Dense(latent_size, name='dense_in_lat')

        self.layer_attn, self.layer_lstm, self.layer_dense, self.layer_dense_lat = [], [], [], []
        for i in range(net_blocks):
            if self.net_attn: self.layer_attn.append(util.MultiHeadAttention(num_heads=num_heads, latent_size=latent_size, memory_size=memory_size, name='attn_{:02d}'.format(i)))
            if self.net_lstm: self.layer_lstm.append(tf.keras.layers.LSTM(mid, activation=util.EvoNormS0(evo), use_bias=False, return_sequences=True, stateful=True, name='lstm_{:02d}'.format(i)))
            else: self.layer_dense.append(tf.keras.layers.Dense(mid, activation=util.EvoNormS0(evo), use_bias=False, name='dense_{:02d}'.format(i)))
            self.layer_dense_lat.append(tf.keras.layers.Dense(latent_size, name='dense_lat_{:02d}'.format(i)))

        self.net_outs, params_size, self.dist, self.logits_step_shape, arch_out = len(spec_out), [], [], [], "O"
        for i in range(self.net_outs):
            num_components, event_shape = spec_out[i]['num_components'], spec_out[i]['event_shape']
            if force_det_out:
                params_size.append(util.Deterministic.params_size(event_shape)); self.dist.append(util.Deterministic(event_shape)); typ = 'p'
            elif not force_cont and spec_out[i]['is_discrete']:
                params_size.append(util.Categorical.params_size(num_components, event_shape)); self.dist.append(util.Categorical(num_components, event_shape)); typ = 'd'
            else:
                num_components *= mixture_size
                params_size.append(util.MixtureLogistic.params_size(num_components, event_shape)); self.dist.append(util.MixtureLogistic(num_components, event_shape)); typ = 'cm'
                # params_size.append(tfp.layers.MixtureLogistic.params_size(num_components, event_shape)); self.dist.append(tfp.layers.MixtureLogistic(num_components, event_shape)) # makes NaNs
                # params_size.append(util.MixtureMultiNormalTriL.params_size(num_components, event_shape, matrix_size=2)); self.dist.append(util.MixtureMultiNormalTriL(num_components, event_shape, matrix_size=2)); typ = 'ct'
            self.logits_step_shape.append(tf.TensorShape([1]+[params_size[i]]))
            arch_out += "{}{}".format(typ, num_components)

        self.layer_dense_logits_out = []
        for i in range(self.net_outs):
            if spec_out[i]['net_type'] == 0: self.layer_dense_logits_out.append(tf.keras.layers.Dense(params_size[i], name='dense_logits_out_{:02d}'.format(i)))

        self.call = tf.function(self.call, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
        self.net_arch = "{}[inD{}-{:02d}{}{}D{}-{}{}]".format(name, inp, net_blocks, ('AT+' if self.net_attn else ''), ('LS+' if self.net_lstm else ''), mid, arch_out, ('-hds'+str(num_heads) if self.net_attn else ''))

    def reset_states(self):
        for layer in self.layer_attn: layer.reset_states()
        for layer in self.layer_lstm: layer.reset_states()
    def call(self, inputs, training=None):
        out = tf.cast(inputs['obs'], self.compute_dtype)
        out = self.layer_flatten(out)
        out = self.layer_dense_in(out)
        out = self.layer_dense_in_lat(out)
        
        for i in range(self.net_blocks):
            out = tf.expand_dims(out, axis=0)
            if self.net_attn: out = self.layer_attn[i](out, training=training)
            if self.net_lstm: out = self.layer_lstm[i](out, training=training)
            out = tf.squeeze(out, axis=0)
            if not self.net_lstm: out = self.layer_dense[i](out)
            out = self.layer_dense_lat[i](out)

        # TODO how to loop through different output layer structures?
        out_logits = [None]*self.net_outs
        for i in range(self.net_outs):
            out_logits[i] = self.layer_dense_logits_out[i](out)

        isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(out), tf.math.is_inf(out)))
        if isinfnan > 0: tf.print('action net out:', out)
        return out_logits


class ValueNet(tf.keras.Model):
    def __init__(self, name, latent_size, net_blocks, net_attn, net_lstm, num_heads=2, memory_size=1):
        super(ValueNet, self).__init__()
        inp, mid, evo = latent_size*2, latent_size*2, int(latent_size/2)
        self.net_blocks, self.net_attn, self.net_lstm = net_blocks, net_attn, net_lstm
        self.layer_flatten = tf.keras.layers.Flatten()

        self.layer_dense_in = tf.keras.layers.Dense(inp, activation=util.EvoNormS0(evo), use_bias=False, name='dense_in')
        self.layer_dense_in_lat = tf.keras.layers.Dense(latent_size, name='dense_in_lat')

        self.layer_attn, self.layer_lstm, self.layer_dense, self.layer_dense_lat = [], [], [], []
        for i in range(net_blocks):
            if self.net_attn: self.layer_attn.append(util.MultiHeadAttention(num_heads=num_heads, latent_size=latent_size, memory_size=memory_size, name='attn_{:02d}'.format(i)))
            if self.net_lstm: self.layer_lstm.append(tf.keras.layers.LSTM(mid, activation=util.EvoNormS0(evo), use_bias=False, return_sequences=True, stateful=True, name='lstm_{:02d}'.format(i)))
            else: self.layer_dense.append(tf.keras.layers.Dense(mid, activation=util.EvoNormS0(evo), use_bias=False, name='dense_{:02d}'.format(i)))
            self.layer_dense_lat.append(tf.keras.layers.Dense(latent_size, name='dense_lat_{:02d}'.format(i)))

        self.layer_dense_out = tf.keras.layers.Dense(1, name='dense_out')

        self.call = tf.function(self.call, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
        self.net_arch = "{}[inD{}-{:02d}{}{}D{}{}]".format(name, inp, net_blocks, ('AT+' if self.net_attn else ''), ('LS+' if self.net_lstm else ''), mid, ('-hds'+str(num_heads) if self.net_attn else ''))

    def reset_states(self):
        for layer in self.layer_attn: layer.reset_states()
        for layer in self.layer_lstm: layer.reset_states()
    def call(self, inputs, training=None):
        out = tf.cast(inputs['obs'], self.compute_dtype)
        out = self.layer_flatten(out)
        out = self.layer_dense_in(out)
        out = self.layer_dense_in_lat(out)
        
        for i in range(self.net_blocks):
            out = tf.expand_dims(out, axis=0)
            if self.net_attn: out = self.layer_attn[i](out, training=training)
            if self.net_lstm: out = self.layer_lstm[i](out, training=training)
            out = tf.squeeze(out, axis=0)
            if not self.net_lstm: out = self.layer_dense[i](out)
            out = self.layer_dense_lat[i](out)

        out = self.layer_dense_out(out)
        return out


class GeneralAI(tf.keras.Model):
    def __init__(self, arch, env, env_render, max_episodes, max_steps, learn_rate, entropy_contrib, returns_disc, force_cont_obs, force_cont_action, latent_size, latent_dist, memory_size):
        super(GeneralAI, self).__init__()
        compute_dtype = tf.dtypes.as_dtype(self.compute_dtype)
        self.float_maxroot = tf.constant(tf.math.sqrt(compute_dtype.max), compute_dtype)
        self.float_eps = tf.constant(tf.experimental.numpy.finfo(compute_dtype).eps, compute_dtype)
        self.compute_zero, self.int32_zero, self.float64_zero = tf.constant(0, compute_dtype), tf.constant(0, tf.int32), tf.constant(0, tf.float64)

        self.arch, self.env, self.env_render, self.force_cont_obs, self.force_cont_action = arch, env, env_render, force_cont_obs, force_cont_action
        self.max_episodes, self.max_steps, self.entropy_contrib, self.returns_disc = tf.constant(max_episodes, tf.int32), tf.constant(max_steps, tf.int32), tf.constant(entropy_contrib, compute_dtype), tf.constant(returns_disc, tf.float64)
        self.dist_prior = tfp.distributions.Independent(tfp.distributions.Logistic(loc=tf.zeros(latent_size, dtype=self.compute_dtype), scale=10.0), reinterpreted_batch_ndims=1)
        # self.dist_prior = tfp.distributions.Independent(tfp.distributions.Uniform(low=tf.cast(tf.fill(latent_size,-10), dtype=self.compute_dtype), high=10), reinterpreted_batch_ndims=1)

        self.obs_spec, self.obs_zero, _ = util.gym_get_spec(env.observation_space, self.compute_dtype, force_cont=force_cont_obs)
        self.action_spec, _, self.action_zero_out = util.gym_get_spec(env.action_space, self.compute_dtype, force_cont=force_cont_action)
        self.obs_spec_len, self.action_spec_len = len(self.obs_spec), len(self.action_spec)
        self.gym_step_shapes = [feat['step_shape'] for feat in self.obs_spec] + [tf.TensorShape((1,1)), tf.TensorShape((1,1))]
        self.gym_step_dtypes = [feat['dtype'] for feat in self.obs_spec] + [tf.float64, tf.bool]
        self.rewards_zero, self.dones_zero = tf.constant([[0]],tf.float64), tf.constant([[False]],tf.bool)
        # self.reward_spec = {'dtype':compute_dtype, 'event_shape':(latent_size,), 'num_components':0}
        # self.done_spec = {'dtype':compute_dtype, 'event_shape':(latent_size,), 'num_components':0}

        if latent_dist == 0: latent_spec = {'dtype':compute_dtype, 'event_shape':(latent_size,), 'num_components':0}
        if latent_dist == 1: latent_spec = {'dtype':compute_dtype, 'event_shape':(latent_size, latent_size), 'num_components':0}
        if latent_dist == 2: latent_spec = {'dtype':compute_dtype, 'event_shape':(latent_size,), 'num_components':latent_size}
        self.latent_spec = latent_spec

        inputs = {'obs':self.obs_zero, 'rewards':self.rewards_zero, 'dones':self.dones_zero}
        if arch in ('PG','AC','TRANS','MU','TEST'):
            self.rep = RepNet('RN', self.obs_spec, latent_spec, latent_dist, latent_size, net_blocks=0, net_attn=False, net_lstm=False, num_heads=2, memory_size=memory_size)
            outputs = self.rep(inputs)
            rep_dist = self.rep.dist(outputs)
            smpl = rep_dist.sample()
            smpl = tf.zeros_like(smpl, latent_spec['dtype'])
            self.latent_zero = smpl
            inputs['obs'] = self.latent_zero

        if arch in ('TEST',):
            self.gen = GenNet('GN', [latent_spec], self.obs_spec, force_cont_obs, latent_size, net_blocks=1, net_attn=False, net_lstm=False, num_heads=2, memory_size=memory_size); outputs = self.gen(inputs)

        self.action = GenNet('AN', self.action_spec, force_cont_action, latent_size, net_blocks=1, net_attn=False, net_lstm=False, num_heads=2, memory_size=memory_size, force_det_out=False); outputs = self.action(inputs)
        if arch in ('AC','MU'): self.value = ValueNet('VN', latent_size, net_blocks=1, net_attn=False, net_lstm=False, num_heads=2, memory_size=memory_size); outputs = self.value(inputs)

        if arch in ('TRANS','MU','TEST'):
            inputs['actions'] = self.action_zero_out
            self.trans = TransNet('TN', self.action_spec, latent_spec, latent_dist, latent_size, net_blocks=2, net_attn=True, net_lstm=False, num_heads=2, memory_size=memory_size); outputs = self.trans(inputs)
        if arch in ('MU','TEST'):
            self.rwd = RewardNet('RWD'); outputs = self.rwd(inputs)
            self.done = DoneNet('DON'); outputs = self.done(inputs)
            # self.rwd = GenNet('RWD', self.reward_spec, force_cont_action, latent_size); outputs = self.rwd(inputs)
            # self.done = GenNet('DON', self.done_spec, force_cont_action, latent_size); outputs = self.done(inputs)

        self.discounted_sum = tf.Variable(0, dtype=tf.float64, trainable=False)
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate, epsilon=self.float_eps)


        metrics = {'rewards_total':np.float64,'rewards_final':np.float64,'steps':np.int64}
        metrics_loss = OrderedDict()
        if arch not in ('MU',): metrics_loss['totals'] = {'loss_total':np.float64}
        # if arch in ('AC','MU'):
        if arch in ('AC','MU'): metrics_loss['nets'] = {'loss_action':np.float64,'loss_value':np.float64}
        if arch in ('MU','TEST'):
            metrics_loss['nets2'] = {'loss_policy':np.float64,'loss_return':np.float64}
            metrics_loss['nets1'] = {'loss_rwd':np.float64,'loss_done':np.float64}
        # if arch in ('PG','MU'): metrics_loss['extras'] = {'returns':np.float64}
        if arch in ('AC','MU','TEST'): metrics_loss['extras'] = {'returns':np.float64,'advantages':np.float64}
        # if arch in ('MU','TEST'):
        #     metrics_loss['nets3'] = {'loss_total_img':np.float64,'returns_img':np.float64}
        #     # metrics_loss['extras1'] = {'steps_img':np.float64,'':np.float64}
        #     metrics_loss['extras1'] = {'steps_img':np.float64}

        for k in metrics.keys(): metrics[k] = np.zeros((max_episodes), metrics[k])
        for loss_group in metrics_loss.values():
            for k in loss_group.keys(): loss_group[k] = [[] for i in range(max_episodes)]
        self.metrics_main, self.metrics_loss = metrics, metrics_loss
        
        # TF bug that wont set graph options with tf.function decorator inside a class
        self.reset_states = tf.function(self.reset_states, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
        self.reset_states()
        arch_run = getattr(self, arch); arch_run = tf.function(arch_run, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS); setattr(self, arch, arch_run)


    def metrics_update(self, *args):
        args = list(args)
        for i in range(len(args)): args[i] = args[i].item()
        episode = args[0]
        self.metrics_main['rewards_total'][episode] += args[1]
        self.metrics_main['rewards_final'][episode] = args[2]
        self.metrics_main['steps'][episode] += args[3]
        if args[4] is not False and 'totals' in self.metrics_loss: self.metrics_loss['totals']['loss_total'][episode].append(args[4])
        if args[5] is not False and 'nets' in self.metrics_loss: self.metrics_loss['nets']['loss_action'][episode].append(args[5])
        if args[6] is not False and 'nets' in self.metrics_loss: self.metrics_loss['nets']['loss_value'][episode].append(args[6])
        if args[7] is not False and 'extras' in self.metrics_loss: self.metrics_loss['extras']['returns'][episode].append(args[7])
        if args[8] is not False and 'extras' in self.metrics_loss: self.metrics_loss['extras']['advantages'][episode].append(args[8])
        if args[9] is not False and 'nets1' in self.metrics_loss: self.metrics_loss['nets1']['loss_rwd'][episode].append(args[9])
        if args[10] is not False and 'nets1' in self.metrics_loss: self.metrics_loss['nets1']['loss_done'][episode].append(args[10])
        if args[11] is not False and 'nets2' in self.metrics_loss: self.metrics_loss['nets2']['loss_policy'][episode].append(args[11])
        if args[12] is not False and 'nets2' in self.metrics_loss: self.metrics_loss['nets2']['loss_return'][episode].append(args[12])
        if args[13] is not False and 'nets3' in self.metrics_loss: self.metrics_loss['nets3']['loss_total_img'][episode].append(args[13])
        if args[14] is not False and 'nets3' in self.metrics_loss: self.metrics_loss['nets3']['returns_img'][episode].append(args[14])
        if args[15] is not False and 'extras1' in self.metrics_loss: self.metrics_loss['extras1']['steps_img'][episode].append(args[15])
        if args[16] is not False and 'extras1' in self.metrics_loss: self.metrics_loss['extras1'][''][episode].append(args[16])
        return np.asarray(0, np.int32) # dummy


    # TODO use ZMQ for remote messaging, latent pooling
    def env_reset(self, dummy):
        obs, reward, done = self.env.reset(), 0.0, False
        if self.env_render: self.env.render()
        if hasattr(self.env,'np_struc'): rtn = util.gym_struc_to_feat(obs)
        else: rtn = util.gym_space_to_feat(obs, self.env.observation_space)
        rtn += [np.asarray([[reward]], np.float64), np.asarray([[done]], bool)]
        return rtn
    def env_step(self, *args): # args = tuple of ndarrays
        if hasattr(self.env,'np_struc'): action = util.gym_out_to_struc(list(args), self.env.action_dtype)
        else: action = util.gym_out_to_space(args, self.env.action_space, [0])
        obs, reward, done, _ = self.env.step(action)
        if self.env_render: self.env.render()
        if hasattr(self.env,'np_struc'): rtn = util.gym_struc_to_feat(obs)
        else: rtn = util.gym_space_to_feat(obs, self.env.observation_space)
        rtn += [np.asarray([[reward]], np.float64), np.asarray([[done]], bool)]
        return rtn


    def reset_states(self):
        for net in self.layers:
            if hasattr(net, 'reset_states'): net.reset_states()


    def loss_diff(self, out, targets=None): # deterministic difference
        if isinstance(out, list):
            loss = self.compute_zero
            for i in range(len(out)):
                o, t = tf.cast(out[i], self.compute_dtype), tf.cast(targets[i], self.compute_dtype)
                # loss = loss + tf.math.abs(tf.math.subtract(o, t)) # MAE
                loss = loss + tf.math.square(tf.math.subtract(o, t)) # MSE
        else:
            out = tf.cast(out, self.compute_dtype)
            if targets is None: diff = out
            else:
                targets = tf.cast(targets, self.compute_dtype)
                diff = tf.math.subtract(out, targets)
            # loss = tf.where(tf.math.less(diff, self.compute_zero), tf.math.negative(diff), diff) # MAE
            # loss = tf.math.abs(diff) # MAE
            loss = tf.math.square(diff) # MSE
        loss = tf.math.reduce_sum(loss, axis=tf.range(1, tf.rank(loss)))
        return loss

    def loss_likelihood(self, dist, targets):
        if isinstance(dist, list):
            loss = self.compute_zero
            for i in range(len(dist)):
                t = tf.cast(targets[i], dist[i].dtype)
                loss = loss - dist[i].log_prob(t)
        else:
            targets = tf.cast(targets, dist.dtype)
            loss = -dist.log_prob(targets)

        isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(loss), tf.math.is_inf(loss)))
        if isinfnan > 0: tf.print('NaN/Inf likelihood loss:', loss)
        return loss

    def loss_bound(self, dist, targets):
        loss = -self.loss_likelihood(dist, targets)
        # if not self.categorical:
        # loss = loss - self.dist_prior.log_prob(targets)

        isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(loss), tf.math.is_inf(loss)))
        if isinfnan > 0: tf.print('NaN/Inf bound loss:', loss)
        return loss

    def loss_PG(self, dist, targets, returns): # policy gradient, actor/critic
        loss = self.loss_likelihood(dist, targets)
        returns = tf.cast(returns, self.compute_dtype)
        returns = tf.squeeze(returns, axis=-1)
        loss = loss * returns # * 1e-2
        # loss = loss * tf.math.softplus(returns)
        # if isinstance(dist, list):
        #     entropy = self.compute_zero
        #     for i in range(len(dist)): entropy = entropy + dist[i].entropy()
        # else: entropy = dist.entropy()
        # loss = loss - entropy * self.entropy_contrib # "Soft Actor Critic" = try increase entropy

        isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(loss), tf.math.is_inf(loss)))
        if isinfnan > 0: tf.print('NaN/Inf PG loss:', loss)
        return loss



    def PG_actor(self, inputs):
        print("tracing -> GeneralAI PG_actor")
        obs, actions = [None]*self.obs_spec_len, [None]*self.action_spec_len
        for i in range(self.obs_spec_len): obs[i] = tf.TensorArray(self.obs_spec[i]['dtype'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.obs_spec[i]['event_shape'])
        for i in range(self.action_spec_len): actions[i] = tf.TensorArray(self.action_spec[i]['dtype_out'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.action_spec[i]['event_shape'])
        rewards = tf.TensorArray(tf.float64, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        dones = tf.TensorArray(tf.bool, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        returns = tf.TensorArray(tf.float64, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

        step = tf.constant(0)
        while step < self.max_steps and not inputs['dones'][-1][0]:
            for i in range(self.obs_spec_len): obs[i] = obs[i].write(step, inputs['obs'][i][-1])
            returns = returns.write(step, [self.float64_zero])

            rep_logits = self.rep(inputs); rep_dist = self.rep.dist(rep_logits)
            inputs['obs'] = rep_dist.sample()

            action_logits = self.action(inputs)
            action = [None]*self.action_spec_len
            for i in range(self.action_spec_len):
                action_dist = self.action.dist[i](action_logits[i])
                action[i] = action_dist.sample()
                actions[i] = actions[i].write(step, action[i][-1])
                action[i] = util.discretize(action[i], self.action_spec[i], self.force_cont_action)

            np_in = tf.numpy_function(self.env_step, action, self.gym_step_dtypes)
            for i in range(len(np_in)): np_in[i].set_shape(self.gym_step_shapes[i])
            inputs['obs'], inputs['rewards'], inputs['dones'] = np_in[:-2], np_in[-2], np_in[-1]

            rewards = rewards.write(step, inputs['rewards'][-1])
            dones = dones.write(step, inputs['dones'][-1])
            returns_updt = returns.stack()
            returns_updt = returns_updt + inputs['rewards'][-1]
            returns = returns.unstack(returns_updt)

            step += 1

        outputs = {}
        out_obs, out_actions = [None]*self.obs_spec_len, [None]*self.action_spec_len
        for i in range(self.obs_spec_len): out_obs[i] = obs[i].stack()
        for i in range(self.action_spec_len): out_actions[i] = actions[i].stack()
        outputs['obs'], outputs['actions'], outputs['rewards'], outputs['dones'], outputs['returns'] = out_obs, out_actions, rewards.stack(), dones.stack(), returns.stack()
        return outputs, inputs

    # TODO try one step history learner, training head is max_steps steps behind live
    def PG_learner(self, inputs, training=True):
        print("tracing -> GeneralAI PG_learner")

        rep_logits = self.rep(inputs, training=training); rep_dist = self.rep.dist(rep_logits)
        inputs['obs'] = rep_dist.sample()

        action_logits = self.action(inputs, training=training)
        action_dist = [None]*self.action_spec_len
        for i in range(self.action_spec_len): action_dist[i] = self.action.dist[i](action_logits[i])
        
        loss = {}
        loss['total'] = self.loss_PG(action_dist, inputs['actions'], inputs['returns'])
        return loss

    def PG_run_episode(self, inputs, episode, training=True):
        print("tracing -> GeneralAI PG_run_episode")
        # TODO how unlimited length episodes without sacrificing returns signal?
        while not inputs['dones'][-1][0]:
            outputs, inputs = self.PG_actor(inputs)
            with tf.GradientTape() as tape:
                loss = self.PG_learner(outputs)
            gradients = tape.gradient(loss['total'], self.rep.trainable_variables + self.action.trainable_variables)
            self._optimizer.apply_gradients(zip(gradients, self.rep.trainable_variables + self.action.trainable_variables))

            metrics = [episode, tf.math.reduce_sum(outputs['rewards']), outputs['rewards'][-1][0], tf.shape(outputs['rewards'])[0],
                tf.math.reduce_mean(loss['total']), False, False,
                tf.math.reduce_sum(outputs['returns']), False, False, False, False, False, False, False, False, False]
            dummy = tf.numpy_function(self.metrics_update, metrics, [tf.int32])

    def PG(self):
        print("tracing -> GeneralAI PG")
        for episode in tf.range(self.max_episodes):
            tf.autograph.experimental.set_loop_options(parallel_iterations=1)
            self.reset_states()
            np_in = tf.numpy_function(self.env_reset, [tf.constant(0)], self.gym_step_dtypes)
            for i in range(len(np_in)): np_in[i].set_shape(self.gym_step_shapes[i])
            inputs = {'obs':np_in[:-2], 'rewards':np_in[-2], 'dones':np_in[-1]}
            self.PG_run_episode(inputs, episode)



    def AC_actor(self, inputs):
        print("tracing -> GeneralAI AC_actor")
        obs, actions = [None]*self.obs_spec_len, [None]*self.action_spec_len
        for i in range(self.obs_spec_len): obs[i] = tf.TensorArray(self.obs_spec[i]['dtype'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.obs_spec[i]['event_shape'])
        for i in range(self.action_spec_len): actions[i] = tf.TensorArray(self.action_spec[i]['dtype_out'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.action_spec[i]['event_shape'])
        rewards = tf.TensorArray(tf.float64, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        dones = tf.TensorArray(tf.bool, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        returns = tf.TensorArray(tf.float64, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

        step = tf.constant(0)
        while step < self.max_steps and not inputs['dones'][-1][0]: # max_steps for limiting memory usage
            # tf.autograph.experimental.set_loop_options(parallel_iterations=1)
            # tf.autograph.experimental.set_loop_options(shape_invariants=[(inputs['obs'], [tf.TensorShape([None,None])]), (inputs['rewards'], tf.TensorShape([None,None])), (inputs['dones'], tf.TensorShape([None,None]))])
            # tf.autograph.experimental.set_loop_options(shape_invariants=[(outputs['rewards'], [None,1]), (outputs['dones'], [None,1]), (outputs['returns'], [None,1])])
            for i in range(self.obs_spec_len): obs[i] = obs[i].write(step, inputs['obs'][i][-1])
            returns = returns.write(step, [self.float64_zero])

            rep_logits = self.rep(inputs); rep_dist = self.rep.dist(rep_logits)
            inputs['obs'] = rep_dist.sample()

            action_logits = self.action(inputs)
            action = [None]*self.action_spec_len
            for i in range(self.action_spec_len):
                action_dist = self.action.dist[i](action_logits[i])
                action[i] = action_dist.sample()
                actions[i] = actions[i].write(step, action[i][-1])
                action[i] = util.discretize(action[i], self.action_spec[i], self.force_cont_action)

            np_in = tf.numpy_function(self.env_step, action, self.gym_step_dtypes)
            for i in range(len(np_in)): np_in[i].set_shape(self.gym_step_shapes[i])
            # inputs = {'obs':np_in[:-2], 'rewards':np_in[-2], 'dones':np_in[-1]}
            inputs['obs'], inputs['rewards'], inputs['dones'] = np_in[:-2], np_in[-2], np_in[-1]

            rewards = rewards.write(step, inputs['rewards'][-1])
            dones = dones.write(step, inputs['dones'][-1])
            returns_updt = returns.stack()
            returns_updt = returns_updt + inputs['rewards'][-1]
            returns = returns.unstack(returns_updt)

            step += 1

        # for i in range(self.obs_spec_len): obs[i] = obs[i].write(step, inputs['obs'][i][-1]) # TD loss

        outputs = {}
        out_obs, out_actions = [None]*self.obs_spec_len, [None]*self.action_spec_len
        for i in range(self.obs_spec_len): out_obs[i] = obs[i].stack()
        for i in range(self.action_spec_len): out_actions[i] = actions[i].stack()
        outputs['obs'], outputs['actions'], outputs['rewards'], outputs['dones'], outputs['returns'] = out_obs, out_actions, rewards.stack(), dones.stack(), returns.stack()
        return outputs, inputs

    def AC_learner(self, inputs, inputs_last, training=True):
        print("tracing -> GeneralAI AC_learner")

        rep_logits = self.rep(inputs, training=training); rep_dist = self.rep.dist(rep_logits)
        inputs['obs'] = rep_dist.sample()

        values = self.value(inputs, training=training)
        # value_logits = self.value(inputs, training=training); value_dist = self.value.dist(value_logits)
        # values = value_dist.sample()

        # PG loss
        returns = tf.cast(inputs['returns'], self.compute_dtype)
        advantages = returns - values # new chance of chosen action: if over predict = push away, if under predict = push closer, if can predict = stay

        # # TD loss
        # rewards = tf.squeeze(inputs['rewards'], axis=-1)
        # rewards = tf.cast(rewards, self.compute_dtype)

        # values_next = tf.roll(values, -1, axis=0)
        # if inputs_last['dones'][-1][0]: values_next = tf.concat((values_next[:-2], [self.compute_zero]), axis=0)
        # else: values_next = values_next[:-1]
        # values = values[:-1]
        # inputs['obs'] = inputs['obs'][:-1]

        # current_q = values
        # target_q = rewards + values_next
        # advantages = target_q - current_q
        # # advantages = 0.1 * current_q + tf.math.abs(advantages)
        # advantages = 0.1 * current_q + tf.math.pow(advantages, 2) # DQNReg

        action_logits = self.action(inputs, training=training)
        action_dist = [None]*self.action_spec_len
        for i in range(self.action_spec_len): action_dist[i] = self.action.dist[i](action_logits[i])
        
        loss = {}
        loss['action'] = self.loss_PG(action_dist, inputs['actions'], advantages)
        loss['value'] = self.loss_diff(advantages)
        # loss['value'] = self.loss_likelihood(value_dist, inputs['returns'])
        loss['total'] = loss['action'] + loss['value']

        loss['advantages'] = advantages
        return loss

    def AC_run_episode(self, inputs, episode, training=True):
        print("tracing -> GeneralAI AC_run_episode")
        while not inputs['dones'][-1][0]:
            # tf.autograph.experimental.set_loop_options(parallel_iterations=1)
            outputs, inputs = self.AC_actor(inputs)
            with tf.GradientTape() as tape:
                loss = self.AC_learner(outputs, inputs)
            gradients = tape.gradient(loss['total'], self.rep.trainable_variables + self.action.trainable_variables + self.value.trainable_variables)
            # isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(gradients[0]), tf.math.is_inf(gradients[0])))
            # if isinfnan > 0: tf.print('\ngradients', gradients[0]); break
            self._optimizer.apply_gradients(zip(gradients, self.rep.trainable_variables + self.action.trainable_variables + self.value.trainable_variables))

            metrics = [episode, tf.math.reduce_sum(outputs['rewards']), outputs['rewards'][-1][0], tf.shape(outputs['rewards'])[0],
                tf.math.reduce_mean(loss['total']), tf.math.reduce_mean(loss['action']), tf.math.reduce_mean(loss['value']),
                tf.math.reduce_sum(outputs['returns']), tf.math.reduce_sum(loss['advantages']), False, False, False, False, False, False, False, False]
            dummy = tf.numpy_function(self.metrics_update, metrics, [tf.int32])

    def AC(self):
        print("tracing -> GeneralAI AC")
        for episode in tf.range(self.max_episodes):
            tf.autograph.experimental.set_loop_options(parallel_iterations=1) # TODO parallel wont work with single instance env, will this work multiple?
            self.reset_states()
            np_in = tf.numpy_function(self.env_reset, [tf.constant(0)], self.gym_step_dtypes)
            for i in range(len(np_in)): np_in[i].set_shape(self.gym_step_shapes[i])
            inputs = {'obs':np_in[:-2], 'rewards':np_in[-2], 'dones':np_in[-1]}
            self.AC_run_episode(inputs, episode)



    def TRANS_actor(self, inputs):
        print("tracing -> GeneralAI TRANS_actor")
        obs = [None]*self.obs_spec_len
        for i in range(self.obs_spec_len): obs[i] = tf.TensorArray(self.obs_spec[i]['dtype'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.obs_spec[i]['event_shape'])
        actions = [None]*self.action_spec_len
        for i in range(self.action_spec_len): actions[i] = tf.TensorArray(self.action_spec[i]['dtype_out'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.action_spec[i]['event_shape'])
        # latents_next = tf.TensorArray(self.latent_spec['dtype'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.latent_spec['event_shape'])
        rewards = tf.TensorArray(tf.float64, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        targets = [None]*self.obs_spec_len
        for i in range(self.obs_spec_len): targets[i] = tf.TensorArray(self.obs_spec[i]['dtype'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.obs_spec[i]['event_shape'])

        # inputs_rep = {'obs':self.latent_zero}
        inputs_rep = {'obs':self.latent_zero, 'actions':self.action_zero_out}
        step = tf.constant(0)
        while step < self.max_steps and not inputs['dones'][-1][0]:
            for i in range(self.obs_spec_len): obs[i] = obs[i].write(step, inputs['obs'][i][-1])
            for i in range(self.action_spec_len): actions[i] = actions[i].write(step, inputs_rep['actions'][i][-1])

            rep_logits = self.rep(inputs); rep_dist = self.rep.dist(rep_logits)
            inputs_rep['obs'] = rep_dist.sample()

            trans_logits = self.trans(inputs_rep); trans_dist = self.trans.dist(trans_logits)
            inputs_rep['obs'] = trans_dist.sample()
            # latents_next = latents_next.write(step, inputs_rep['obs'][-1])

            action_logits = self.action(inputs_rep)
            action, action_dis = [None]*self.action_spec_len, [None]*self.action_spec_len
            for i in range(self.action_spec_len):
                action_dist = self.action.dist[i](action_logits[i])
                action[i] = action_dist.sample()
                # action[i] = tf.cast(action[i], self.action_spec[i]['dtype_out']) # force_det_out
                action_dis[i] = util.discretize(action[i], self.action_spec[i], self.force_cont_action)
            inputs_rep['actions'] = action

            np_in = tf.numpy_function(self.env_step, action_dis, self.gym_step_dtypes)
            for i in range(len(np_in)): np_in[i].set_shape(self.gym_step_shapes[i])
            inputs['obs'], inputs['rewards'], inputs['dones'] = np_in[:-2], np_in[-2], np_in[-1]

            rewards = rewards.write(step, inputs['rewards'][-1])
            for i in range(self.obs_spec_len): targets[i] = targets[i].write(step, inputs['obs'][i][-1])
            step += 1

        outputs = {}
        out_obs = [None]*self.obs_spec_len
        for i in range(self.obs_spec_len): out_obs[i] = obs[i].stack()
        out_actions = [None]*self.action_spec_len
        for i in range(self.action_spec_len): out_actions[i] = actions[i].stack()
        out_targets = [None]*self.obs_spec_len
        for i in range(self.obs_spec_len): out_targets[i] = targets[i].stack()
        # outputs['obs'], outputs['rewards'], outputs['targets'] = latents_next.stack(), rewards.stack(), out_targets
        # outputs['obs'], outputs['rewards'], outputs['targets'] = out_obs, rewards.stack(), out_targets
        outputs['obs'], outputs['actions'], outputs['rewards'], outputs['targets'] = out_obs, out_actions, rewards.stack(), out_targets
        return outputs, inputs

    def TRANS_learner(self, inputs, training=True):
        print("tracing -> GeneralAI TRANS_learner")
        # inputs_rep = {'obs':self.latent_zero}
        inputs_rep = {'obs':self.latent_zero, 'actions':inputs['actions']}

        rep_logits = self.rep(inputs, training=training); rep_dist = self.rep.dist(rep_logits)
        inputs_rep['obs'] = rep_dist.sample()

        trans_logits = self.trans(inputs_rep, training=training); trans_dist = self.trans.dist(trans_logits)
        inputs_rep['obs'] = trans_dist.sample()
        # latents_next = latents_next.write(step, inputs_rep['obs'][-1])

        action_logits = self.action(inputs_rep, training=training)
        action_dist = [None]*self.action_spec_len
        for i in range(self.action_spec_len):
            action_dist[i] = self.action.dist[i](action_logits[i])
            # action_dist[i] = action_dist[i].sample() # force_det_out

        loss = {}
        loss['total'] = self.loss_likelihood(action_dist, inputs['targets'])
        # loss['total'] = self.loss_diff(action_dist, inputs['targets']) # force_det_out
        return loss

    def TRANS_run_episode(self, inputs, episode, training=True):
        print("tracing -> GeneralAI TRANS_run_episode")
        while not inputs['dones'][-1][0]:
            outputs, inputs = self.TRANS_actor(inputs)
            with tf.GradientTape() as tape:
                loss = self.TRANS_learner(outputs)
            gradients = tape.gradient(loss['total'], self.rep.trainable_variables + self.trans.trainable_variables + self.action.trainable_variables)
            self._optimizer.apply_gradients(zip(gradients, self.rep.trainable_variables + self.trans.trainable_variables + self.action.trainable_variables))

            metrics = [episode, tf.math.reduce_sum(outputs['rewards']), outputs['rewards'][-1][0], tf.shape(outputs['rewards'])[0],
                tf.math.reduce_mean(loss['total']), False, False, False, False,
                False, False, False, False, False, False, False, False
            ]
            dummy = tf.numpy_function(self.metrics_update, metrics, [tf.int32])

    def TRANS(self):
        print("tracing -> GeneralAI TRANS")
        for episode in tf.range(self.max_episodes):
            tf.autograph.experimental.set_loop_options(parallel_iterations=1)
            self.reset_states()
            np_in = tf.numpy_function(self.env_reset, [tf.constant(0)], self.gym_step_dtypes)
            for i in range(len(np_in)): np_in[i].set_shape(self.gym_step_shapes[i])
            inputs = {'obs':np_in[:-2], 'rewards':np_in[-2], 'dones':np_in[-1]}
            self.TRANS_run_episode(inputs, episode)



    def MU_imagine(self, inputs):
        print("tracing -> GeneralAI MU_imagine")
        obs = tf.TensorArray(self.latent_spec['dtype'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.latent_spec['event_shape'])
        actions = [None]*self.action_spec_len
        for i in range(self.action_spec_len): actions[i] = tf.TensorArray(self.action_spec[i]['dtype_out'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.action_spec[i]['event_shape'])
        rewards = tf.TensorArray(tf.float64, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        dones = tf.TensorArray(tf.bool, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        returns = tf.TensorArray(tf.float64, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

        inputs_img = {'obs':inputs['obs'], 'actions':self.action_zero_out, 'rewards':self.rewards_zero, 'dones':self.dones_zero}

        step = tf.constant(0)
        while step < self.max_steps and not inputs_img['dones'][-1][0]:
            obs = obs.write(step, inputs_img['obs'][-1])
            returns = returns.write(step, [self.float64_zero])

            action_logits = self.action(inputs_img)
            action = [None]*self.action_spec_len
            for i in range(self.action_spec_len):
                action_dist = self.action.dist[i](action_logits[i])
                action[i] = action_dist.sample()
                actions[i] = actions[i].write(step, action[i][-1])
            inputs_img['actions'] = action

            trans_logits = self.trans(inputs_img); trans_dist = self.trans.dist(trans_logits)
            inputs_img['obs'] = trans_dist.sample()

            rwd_logits = self.rwd(inputs_img); rwd_dist = self.rwd.dist(rwd_logits)
            done_logits = self.done(inputs_img); done_dist = self.done.dist(done_logits)
            inputs_img['rewards'], inputs_img['dones'] = tf.cast(rwd_dist.sample(), tf.float64), tf.cast(done_dist.sample(), tf.bool)

            rewards = rewards.write(step, inputs_img['rewards'][-1])
            dones = dones.write(step, inputs_img['dones'][-1])
            returns_updt = returns.stack()
            returns_updt = returns_updt + inputs_img['rewards'][-1][0]
            returns = returns.unstack(returns_updt)

            step += 1

        outputs = {}
        out_actions = [None]*self.action_spec_len
        for i in range(self.action_spec_len): out_actions[i] = actions[i].stack()
        outputs['obs'], outputs['actions'], outputs['rewards'], outputs['dones'], outputs['returns'] = obs.stack(), out_actions, rewards.stack(), dones.stack(), returns.stack()
        return outputs

    def MU_img_learner(self, inputs, training=True):
        print("tracing -> GeneralAI MU_img_learner")

        action_logits = self.action(inputs, training=training)
        action_dist = [None]*self.action_spec_len
        for i in range(self.action_spec_len): action_dist[i] = self.action.dist[i](action_logits[i])
        
        loss = {}
        loss['total'] = self.loss_PG(action_dist, inputs['actions'], inputs['returns'])
        return loss

    def MU_actor(self, inputs):
        print("tracing -> GeneralAI MU_actor")
        obs, actions = [None]*self.obs_spec_len, [None]*self.action_spec_len
        for i in range(self.obs_spec_len): obs[i] = tf.TensorArray(self.obs_spec[i]['dtype'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.obs_spec[i]['event_shape'])
        for i in range(self.action_spec_len): actions[i] = tf.TensorArray(self.action_spec[i]['dtype_out'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.action_spec[i]['event_shape'])
        rewards = tf.TensorArray(tf.float64, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        dones = tf.TensorArray(tf.bool, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        returns = tf.TensorArray(tf.float64, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

        step = tf.constant(0)
        while step < self.max_steps and not inputs['dones'][-1][0]:
            for i in range(self.obs_spec_len): obs[i] = obs[i].write(step, inputs['obs'][i][-1])
            returns = returns.write(step, [self.float64_zero])

            rep_logits = self.rep(inputs); rep_dist = self.rep.dist(rep_logits)
            inputs['obs'] = rep_dist.sample()

            outputs_img = self.MU_imagine(inputs)
            with tf.GradientTape() as tape:
                loss_img = self.MU_img_learner(outputs_img)
            gradients = tape.gradient(loss_img['total'], self.action.trainable_variables)
            self._optimizer.apply_gradients(zip(gradients, self.action.trainable_variables))

            action_logits = self.action(inputs)
            action = [None]*self.action_spec_len
            for i in range(self.action_spec_len):
                action_dist = self.action.dist[i](action_logits[i])
                action[i] = action_dist.sample()
                actions[i] = actions[i].write(step, action[i][-1])
                action[i] = util.discretize(action[i], self.action_spec[i], self.force_cont_action)

            np_in = tf.numpy_function(self.env_step, action, self.gym_step_dtypes)
            for i in range(len(np_in)): np_in[i].set_shape(self.gym_step_shapes[i])
            inputs['obs'], inputs['rewards'], inputs['dones'] = np_in[:-2], np_in[-2], np_in[-1]

            rewards = rewards.write(step, inputs['rewards'][-1])
            dones = dones.write(step, inputs['dones'][-1])
            returns_updt = returns.stack()
            returns_updt = returns_updt + inputs['rewards'][-1]
            returns = returns.unstack(returns_updt)

            step += 1

        outputs = {}
        out_obs, out_actions = [None]*self.obs_spec_len, [None]*self.action_spec_len
        for i in range(self.obs_spec_len): out_obs[i] = obs[i].stack()
        for i in range(self.action_spec_len): out_actions[i] = actions[i].stack()
        outputs['obs'], outputs['actions'], outputs['rewards'], outputs['dones'], outputs['returns'] = out_obs, out_actions, rewards.stack(), dones.stack(), returns.stack()
        return outputs, inputs


    def MU_learner(self, inputs, training=True):
        print("tracing -> GeneralAI MU_learner")

        rep_logits = self.rep(inputs, training=training); rep_dist = self.rep.dist(rep_logits)
        inputs['obs'] = rep_dist.sample()

        action_logits = self.action(inputs, training=training)
        action_dist = [None]*self.action_spec_len
        for i in range(self.action_spec_len): action_dist[i] = self.action.dist[i](action_logits[i])

        values = self.value(inputs, training=training)

        returns = tf.cast(inputs['returns'], self.compute_dtype)
        advantages = returns - values

        loss = {}
        loss['action'] = self.loss_PG(action_dist, inputs['actions'], advantages)
        loss['value'] = self.loss_diff(advantages)


        loss_policy = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_return = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_reward = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_done = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

        inputs_img = {'obs':inputs['obs'][0:1], 'actions':self.action_zero_out}

        for step in tf.range(tf.shape(inputs['obs'])[0]):

            # action_dist = [None]*self.action_spec_len
            # for i in range(self.action_spec_len): action_dist[i] = self.action.dist[i](action_logits[i])
            # value_logits = self.value(inputs_img, training=training); value_dist = self.value.dist(value_logits)

            # # loss_policy = loss_policy.write(step, self.loss_likelihood(action_dist, inputs_img['actions']))
            # loss_policy = loss_policy.write(step, self.loss_PG(action_dist, action, inputs['returns'][step:step+1]))
            # # loss_return = loss_return.write(step, self.loss_likelihood(value_dist, inputs['returns'][step:step+1]))


            # action_logits_img = self.action(inputs_img, training=training)
            # action_logits_cur = [None]*self.action_spec_len
            # for i in range(self.action_spec_len):
            #     action_logits_cur[i] = action_logits[i][step:step+1]
            #     action_logits_cur[i].set_shape(self.action.logits_step_shape[i])

            # values_img = self.value(inputs_img, training=training)

            # loss_policy = loss_policy.write(step, self.loss_diff(action_logits_img, action_logits_cur))
            # loss_return = loss_return.write(step, self.loss_diff(values_img, values[step:step+1]))


            action = [None]*self.action_spec_len
            for i in range(self.action_spec_len):
                action[i] = inputs['actions'][i][step:step+1]
                action[i].set_shape(self.action_spec[i]['step_shape'])
            inputs_img['actions'] = action

            trans_logits = self.trans(inputs_img, training=training); trans_dist = self.trans.dist(trans_logits)
            inputs_img['obs'] = trans_dist.sample()

            rwd_logits = self.rwd(inputs_img, training=training); rwd_dist = self.rwd.dist(rwd_logits)
            done_logits = self.done(inputs_img, training=training); done_dist = self.done.dist(done_logits)

            loss_reward = loss_reward.write(step, self.loss_likelihood(rwd_dist, inputs['rewards'][step:step+1]))
            loss_done = loss_done.write(step, self.loss_likelihood(done_dist, inputs['dones'][step:step+1]))

        loss['policy'], loss['return'], loss['reward'], loss['done'] = loss_policy.concat(), loss_return.concat(), loss_reward.concat(), loss_done.concat()
        # loss['total'] = loss['policy'] + loss['return'] + loss['reward'] + loss['done']
        loss['total'] = loss['action'] + loss['value'] + loss['policy'] + loss['return'] + loss['reward'] + loss['done']
        loss['advantages'] = advantages
        return loss


    def MU_run_episode(self, inputs, episode, training=True):
        print("tracing -> GeneralAI MU_run_episode")
        while not inputs['dones'][-1][0]:

            outputs, inputs = self.MU_actor(inputs)
            with tf.GradientTape() as tape:
                loss = self.MU_learner(outputs)
            gradients = tape.gradient(loss['total'], self.rep.trainable_variables + self.trans.trainable_variables + self.rwd.trainable_variables + self.done.trainable_variables + self.action.trainable_variables + self.value.trainable_variables)
            self._optimizer.apply_gradients(zip(gradients, self.rep.trainable_variables + self.trans.trainable_variables + self.rwd.trainable_variables + self.done.trainable_variables + self.action.trainable_variables + self.value.trainable_variables))

            metrics = [episode, tf.math.reduce_sum(outputs['rewards']), outputs['rewards'][-1][0], tf.shape(outputs['rewards'])[0],
                tf.math.reduce_mean(loss['total']), tf.math.reduce_mean(loss['action']), tf.math.reduce_mean(loss['value']), tf.math.reduce_sum(outputs['returns']), tf.math.reduce_sum(loss['advantages']), tf.math.reduce_mean(loss['reward']), tf.math.reduce_mean(loss['done']),
                # tf.math.reduce_mean(loss_img['total']), tf.math.reduce_mean(outputs_img['returns']), tf.shape(outputs_img['rewards'])[0], False]
                tf.math.reduce_mean(loss['policy']), tf.math.reduce_mean(loss['return']), False, False, False, False]
            dummy = tf.numpy_function(self.metrics_update, metrics, [tf.int32])

    def MU(self):
        print("tracing -> GeneralAI MU")
        for episode in tf.range(self.max_episodes):
            tf.autograph.experimental.set_loop_options(parallel_iterations=1)
            self.reset_states()
            np_in = tf.numpy_function(self.env_reset, [tf.constant(0)], self.gym_step_dtypes)
            for i in range(len(np_in)): np_in[i].set_shape(self.gym_step_shapes[i])
            inputs = {'obs':np_in[:-2], 'rewards':np_in[-2], 'dones':np_in[-1]}
            self.MU_run_episode(inputs, episode)




def params(): pass
load_model, save_model = False, False
max_episodes = 2000
learn_rate = 1e-5
entropy_contrib = 1e-8
returns_disc = 1.0
force_cont_obs, force_cont_action = False, False
latent_size = 128
latent_dist = 0 # 0 = deterministic, 1 = categorical, 2 = continuous
attn_mem_multi = 1

device_type = 'GPU' # use GPU for large networks or big data
device_type = 'CPU'

machine, device, extra = 'dev', 0, '' # _mixlog-abs-Nentropy3-log1p-Nreparam_obs-tsBoxF-dataBoxI_round

env_async, env_async_clock, env_async_speed = False, 0.001, 1000.0
# env_name, max_steps, env_render, env = 'CartPole', 256, False, gym.make('CartPole-v0'); env.observation_space.dtype = np.dtype('float64')
# env_name, max_steps, env_render, env = 'CartPole', 512, False, gym.make('CartPole-v1'); env.observation_space.dtype = np.dtype('float64')
# env_name, max_steps, env_render, env = 'LunarLand', 1024, False, gym.make('LunarLander-v2')
# env_name, max_steps, env_render, env = 'Copy', 256, False, gym.make('Copy-v0')
# env_name, max_steps, env_render, env = 'Qbert', 1024, False, gym.make('QbertNoFrameskip-v4') # max_steps 400000

# env_name, max_steps, env_render, env = 'LunarLandCont', 1024, False, gym.make('LunarLanderContinuous-v2')
# import envs_local.bipedal_walker as env_; env_name, max_steps, env_render, env = 'BipedalWalker', 2048, False, env_.BipedalWalker()

# import envs_local.random_env as env_; env_name, max_steps, env_render, env = 'TestRnd', 16, False, env_.RandomEnv(True)
import envs_local.data_env as env_; env_name, max_steps, env_render, env = 'DataShkspr', 64, False, env_.DataEnv('shkspr')
# # import envs_local.data_env as env_; env_name, max_steps, env_render, env = 'DataMnist', 64, False, env_.DataEnv('mnist')
# import gym_trader; env_name, max_steps, env_render, env = 'Trader2', 4096, False, gym.make('Trader-v0', agent_id=device, env=1, speed=env_async_speed)

max_steps = 1 # max replay buffer or train interval or bootstrap

# arch = 'TEST' # testing architechures
# arch = 'PG' # Policy Gradient agent, PG loss
# arch = 'AC' # Actor Critic, PG and advantage loss
arch = 'TRANS' # learned Transition dynamics, autoregressive likelihood loss
# arch = 'MU' # Dreamer/planner w/imagination (DeepMind MuZero)
# arch = 'DREAM' # full World Model w/imagination (DeepMind Dreamer)

if __name__ == '__main__':
    # TODO add keyboard control so can stop

    ## manage multiprocessing
    # # setup ctrl,data,param sharing
    # # start agents (real+dreamers)
    # agent = Agent(model)
    # # agent_process = mp.Process(target=agent.vivify, name='AGENT', args=(lock_print, process_ctrl, weights_shared))
    # # agent_process.start()
    # # quit on keyboard (space = save, esc = no save)
    # process_ctrl.value = 0
    # agent_process.join()

    if env_async: import envs_local.async_wrapper as envaw_; env_name, env = env_name+'-asyn', envaw_.AsyncWrapperEnv(env, env_async_clock, env_async_speed, env_render)
    with tf.device("/device:{}:{}".format(device_type,device)):
        model = GeneralAI(arch, env, env_render, max_episodes, max_steps, learn_rate, entropy_contrib, returns_disc, force_cont_obs, force_cont_action, latent_size, latent_dist, memory_size=max_steps*attn_mem_multi)
        name = "gym-{}-{}-{}".format(arch, env_name, ['Ldet','Lcat','Lcon'][latent_dist])
        
        ## debugging
        # model.build(()); model.action.summary(); quit(0)
        # inputs = {'obs':model.obs_zero, 'rewards':tf.constant([[0]],tf.float64), 'dones':tf.constant([[False]],tf.bool)}
        # # inp_sig = [[[tf.TensorSpec(shape=None, dtype=tf.float32)], tf.TensorSpec(shape=None, dtype=tf.float64), tf.TensorSpec(shape=None, dtype=tf.bool)]]
        # # model.AC_actor = tf.function(model.AC_actor, input_signature=inp_sig, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
        # model.AC_actor = tf.function(model.AC_actor, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
        # self.AC_actor = tf.function(self.AC_actor)
        # print(tf.autograph.to_code(model.MU_run_episode, recursive=True, experimental_optional_features=tf.autograph.experimental.Feature.LISTS)); quit(0)
        # # print(tf.autograph.to_code(model.AC_actor.python_function, experimental_optional_features=tf.autograph.experimental.Feature.LISTS)); quit(0)
        # print(model.AC_actor.get_concrete_function(inputs)); quit(0)
        # print(model.AC_actor.get_concrete_function(inputs).graph.as_graph_def()); quit(0)
        # obs, reward, done = env.reset(), 0.0, False
        # # test = model.AC_actor.python_function(inputs)
        # test = model.AC_actor(inputs)
        # print(test); quit(0)


        ## load models
        model_files, name_arch = {}, ""
        for net in model.layers:
            model_name = "{}-{}-a{}".format(net.net_arch, machine, device)
            model_file = "{}/tf-data-models-local/{}.h5".format(curdir, model_name); loaded_model = False
            model_files[net.name] = model_file
            if load_model and tf.io.gfile.exists(model_file):
                net.load_weights(model_file, by_name=True, skip_mismatch=True)
                print("LOADED {} weights from {}".format(net.name, model_file)); loaded_model = True
            name_arch += "   {}{}".format(net.net_arch, 'load' if loaded_model else 'new')


        ## run
        arch_run = getattr(model, arch)
        t1_start = time.perf_counter_ns()
        arch_run()
        total_time = (time.perf_counter_ns() - t1_start) / 1e9 # seconds
        env.close()


        ## metrics
        metrics, metrics_loss = model.metrics_main, model.metrics_loss
        for loss_group in metrics_loss.values():
            for k in loss_group.keys():
                for j in range(len(loss_group[k])): loss_group[k][j] = np.mean(loss_group[k][j])
        # TODO np.mean, reduce size if above 100,000-200,000 episodes

        name = "{}-{}-a{}-{}{}".format(name, machine, device, time.strftime("%Y_%m_%d-%H-%M"), extra)
        total_steps = np.sum(metrics['steps'])
        step_time = total_time/total_steps
        title = "{}    [{}-{}] {}\ntime:{}    steps:{}    t/s:{:.8f}     |     lr:{}    dis:{}    en:{}    am:{}    ms:{}".format(name, device_type, tf.keras.backend.floatx(), name_arch, util.print_time(total_time), total_steps, step_time, learn_rate, returns_disc, entropy_contrib, attn_mem_multi, max_steps); print(title)

        import matplotlib as mpl
        mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['blue', 'lightblue', 'green', 'lime', 'red', 'lavender', 'turquoise', 'cyan', 'magenta', 'salmon', 'yellow', 'gold', 'black', 'brown', 'purple', 'pink', 'orange', 'teal', 'coral', 'darkgreen', 'tan'])
        plt.figure(num=name, figsize=(34, 16), tight_layout=True)
        xrng, i, vplts = np.arange(0, max_episodes, 1), 0, 3 + len(metrics_loss)

        rows = 2; plt.subplot2grid((vplts, 1), (i, 0), rowspan=rows); i+=rows; plt.grid(axis='y',alpha=0.3)
        metric_name = 'rewards_total'; metric = np.asarray(metrics[metric_name], np.float64)
        plt.plot(xrng, talib.EMA(metric, timeperiod=max_episodes//10+2), alpha=1.0, label=metric_name); plt.plot(xrng, metric, alpha=0.3)
        metric_name = 'rewards_final'; metric = np.asarray(metrics[metric_name], np.float64)
        plt.plot(xrng, talib.EMA(metric, timeperiod=max_episodes//10+2), alpha=1.0, label=metric_name); plt.plot(xrng, metric, alpha=0.3)
        plt.ylabel('value'); plt.xlabel('episode'); plt.legend(loc='upper left'); plt.title(title)
        rows = 1; plt.subplot2grid((vplts, 1), (i, 0), rowspan=rows); i+=rows; plt.grid(axis='y',alpha=0.3)
        metric_name = 'steps'; metric = np.asarray(metrics[metric_name], np.float64)
        plt.plot(xrng, talib.EMA(metric, timeperiod=max_episodes//10+2), alpha=1.0, label=metric_name); plt.plot(xrng, metric, alpha=0.3)
        plt.ylabel('value'); plt.xlabel('episode'); plt.legend(loc='upper left')

        for loss_group_name, loss_group in metrics_loss.items():
            rows = 1; plt.subplot2grid((vplts, 1), (i, 0), rowspan=rows); i+=rows; plt.grid(axis='y',alpha=0.3)
            for metric_name, metric in loss_group.items():
                metric = np.asarray(metric, np.float64); plt.plot(xrng, talib.EMA(metric, timeperiod=max_episodes//10+2), alpha=1.0, label=metric_name); plt.plot(xrng, metric, alpha=0.3)
            plt.ylabel('value'); plt.xlabel('episode'); plt.legend(loc='upper left')

        plt.show()


        ## save models
        if save_model:
            for net in model.layers:
                model_file = model_files[net.name]
                net.save_weights(model_file)
                print("SAVED {} weights to {}".format(net.name, model_file))
