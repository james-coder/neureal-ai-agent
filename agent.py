"""Agent"""
from collections import OrderedDict
import argparse, os, random, socket, time
from datetime import datetime

import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import gym

import gym_util
import model_util as util
import envs_local.async_wrapper as envaw_
import envs_local.reconfig_wrapper as envrw_
import model_nets as nets  # must be imported after the above tf config

np.set_printoptions(precision=8, suppress=True, linewidth=400, threshold=100)
tf.keras.backend.set_floatx('float64')
# tf.config.run_functions_eagerly(True)
tf.keras.backend.set_epsilon(tf.experimental.numpy.finfo(tf.keras.backend.floatx()).eps) # 1e-7 default
# CUDA 11.8.0, CUDNN 8.6.0.163, tensorflow-gpu==2.9.2, tensorflow_probability==0.17.0

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)



def main():
    """Main"""
    fix_seed = 8675309
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    tf.random.set_seed(fix_seed)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Filter out info initialization logs
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # All logs shown

    machine = socket.gethostname()
    extra = ''
    net_attn = {'net':True, 'io':True, 'out':True, 'ar':True}
    learn_rates = {'action':4e-6} # Policy Gradient agent, PG loss

    parser = argparse.ArgumentParser(description='Neureal AI Agent')
    # use GPU for large networks (over 8 total net blocks?) or output data (512 bytes?)
    parser.add_argument('--device_type', type=str, default='CPU', help='device type options: [CPU, GPU]')
    parser.add_argument('--device_id', type=int, default=0, help='GPU device id')
    parser.add_argument('--gym_make', type=str, default='CartPole-v0', help='OpenAI gym environment: [CartPole-v0, CartPole-v1, LunarLander-v2]')
    parser.add_argument('--load_model', type=bool, default=False, help='load trained model')
    parser.add_argument('--save_model', type=bool, default=False, help='save model after training')
    parser.add_argument('--chkpts', type=int, default=5000, help='number of checkpoints')
    parser.add_argument('--max_episodes', type=int, default=100, help='max episodes')
    parser.add_argument('--latent_size', type=int, default=16, help='latent size')
    parser.add_argument('--latent_dist', type=str, default='d', help="'d' = deterministic, 'c' = categorical, 'mx' = continuous(mixture)")
    parser.add_argument('--mixture_multi', type=int, default=4, help='mixture distribution size,multiply num components')
    parser.add_argument('--net_lstm', type=bool, default=False, help='net lstm')
    parser.add_argument('--aio_max_latents', type=int, default=16, help='aio max latents')
    parser.add_argument('--aug_data_step', type=bool, default=True, help='augment data step')
    parser.add_argument('--aug_data_pos', type=bool, default=True, help='augment data pos')
    parser.add_argument('--trader', type=bool, default=False, help='trader')
    parser.add_argument('--env_async', type=bool, default=False, help='async env wrapper')
    parser.add_argument('--env_async_clock', type=float, default=0.001, help='env async clock')
    parser.add_argument('--env_async_speed', type=float, default=160.0, help='env async speed')
    parser.add_argument('--chart_lim', type=float, default=0.003, help='chart lim')
    parser.add_argument('--arch', type=str, default='PG', help='architecture: [PG]')

    parser.add_argument('--summary', action='store_true', help='Do not train/infer, only output built model summary')

    args = parser.parse_args()
    device_type, device = args.device_type, args.device_id
    load_model, save_model = args.load_model, args.save_model
    chkpts, max_episodes = args.chkpts, args.max_episodes
    latent_size, latent_dist = args.latent_size, args.latent_dist
    mixture_multi, net_lstm = args.mixture_multi, args.net_lstm
    aio_max_latents = args.aio_max_latents
    aug_data_step, aug_data_pos = args.aug_data_step, args.aug_data_pos
    trader = args.trader
    env_async, env_async_clock, env_async_speed = args.env_async, args.env_async_clock, args.env_async_speed
    chart_lim, arch = args.chart_lim, args.arch

    if args.gym_make == 'CartPole-v0':
         # (4) float32    ()2 int64    200  195.0
        env_name, max_steps, env_render, env_reconfig, env = 'CartPole', 256, False, True, gym.make('CartPole-v0')
    elif args.gym_make == 'CartPole-v1':
         # (4) float32    ()2 int64    500  475.0
        env_name, max_steps, env_render, env_reconfig, env = 'CartPole', 512, False, True, gym.make('CartPole-v1')
    elif args.gym_make == 'LunarLander-v2':
         # (8) float32    ()4 int64    1000  200
        env_name, max_steps, env_render, env_reconfig, env = 'LunarLand', 1024, False, True, gym.make('LunarLander-v2')

    if env_async:
        env_name, env = env_name+'-asyn', envaw_.AsyncWrapperEnv(env, env_async_clock, env_async_speed, env_render)
    if env_reconfig:
        env_name, env = env_name+'-r', envrw_.ReconfigWrapperEnv(env)

    with tf.device(f"/device:{device_type}:{device if device_type=='GPU' else 0}"):
        model = GeneralAI(arch, env, trader, env_render, save_model, chkpts, max_episodes, max_steps, learn_rates,
                          latent_size, latent_dist, mixture_multi, net_lstm, net_attn, aio_max_latents, aug_data_step, aug_data_pos)
        name = f"gym-{arch}-{env_name}-{machine}-a{device}{extra}-{time.strftime('%y-%m-%d-%H-%M-%S')}"

        ## load models
        model_files, name_arch = {}, ""
        userdir = os.path.expanduser("~")
        for layer in model.layers:
            model_name = f"{layer.arch_desc}-{machine}-a{device}"
            model_file = f"{userdir}/tf-data-models-local/{model_name}.h5"
            loaded_model = False
            model_files[layer.name] = model_file
            if (load_model or layer.name == 'M') and tf.io.gfile.exists(model_file):
                layer.load_weights(model_file, by_name=True, skip_mismatch=True)
                print(f"LOADED {layer.name} weights from {model_file}")
                loaded_model = True
            name_opt = "-O{}{}".format(
                layer.opt_spec['type'],
                ('' if layer.opt_spec['schedule_type'] == '' else '-S'+layer.opt_spec['schedule_type'])
            ) if hasattr(layer, 'opt_spec') else ''
            name_arch += f"   {layer.arch_desc}{name_opt}-{'load' if loaded_model else 'new'}"
        model.model_files = model_files

        ## run
        print(f"RUN {name}")
        func = getattr(model, arch)
        t1_start = time.perf_counter_ns()
        func()
        total_time = (time.perf_counter_ns() - t1_start) / 1e9 # seconds
        env.close()

        ## metrics
        metrics_loss = model.metrics_loss
        for loss_group in metrics_loss.values():
            for k in loss_group.keys():
                for j in range(len(loss_group[k])):
                    loss_group[k][j] = 0 if loss_group[k][j] == [] else np.mean(loss_group[k][j])

        total_steps = int(np.nansum(metrics_loss['1steps']['steps+']))
        step_time = total_time/total_steps
        learn_rates_txt, attn_txt = "", ""
        for k, v in learn_rates.items():
            learn_rates_txt += f"  {k}:{v:.0e}"
        for k, v in net_attn.items():
            attn_txt += f" {k}" if v else ''
        title = f"{name}    [{device_type}-{tf.keras.backend.floatx()}] {name_arch}\ntime:{util.print_time(total_time)}"
        title += f" steps:{total_steps}    t/s:{step_time:.8f}    ms:{max_steps}"
        title += f"     |     attn:{attn_txt}    al:{aio_max_latents}"
        title += f"     |     a-clk:{env_async_clock}    a-spd:{env_async_speed}    aug:{'S' if aug_data_step else ''}{'P' if aug_data_pos else ''}"
        title += f"     |   {learn_rates_txt}"
        print(title)

        mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[
            'blue', 'lightblue', 'green', 'lime', 'red', 'lavender', 'turquoise', 'cyan', 'magenta', 'salmon', 'yellow',
            'gold', 'black', 'brown', 'purple', 'pink', 'orange', 'teal', 'coral', 'darkgreen', 'tan'
        ])
        plt.figure(num=name, figsize=(34, 18), tight_layout=True)
        xrng, i, vplts = np.arange(0, max_episodes, 1), 0, 0
        for loss_group_name in metrics_loss.keys():
            vplts += int(loss_group_name[0])

        for loss_group_name, loss_group in metrics_loss.items():
            rows, col = int(loss_group_name[0]), 0
            m_min, m_max = [0]*len(loss_group), [0]*len(loss_group)
            combine, yscale = loss_group_name.endswith('*'), ('log' if loss_group_name[1] == '~' else 'linear')
            if combine:
                spg = plt.subplot2grid((vplts, 1), (i, 0), rowspan=rows, xlim=(0, max_episodes), yscale=yscale)
                plt.grid(axis='y', alpha=0.3)
            for metric_name, metric in loss_group.items():
                metric = np.asarray(metric, np.float64)
                m_min[col], m_max[col] = np.nanquantile(metric, chart_lim), np.nanquantile(metric, 1.0-chart_lim)
                if not combine:
                    spg = plt.subplot2grid(
                        (vplts, len(loss_group)), (i, col), rowspan=rows, xlim=(0, max_episodes),
                        ylim=(m_min[col], m_max[col]), yscale=yscale
                    )
                    plt.grid(axis='y', alpha=0.3)
                if metric_name.startswith('-'):
                    plt.plot(xrng, metric, alpha=1.0, label=metric_name)
                else:
                    print(f"max_episodes: {max_episodes}")
                    print(f"metric: {metric}")
                    print(f"metric_name: {metric_name}")
                    print(f"xrng: {xrng}")
                    plt.plot(xrng, util.ewma(metric, window=max_episodes//10+2), alpha=1.0, label=metric_name)
                    plt.plot(xrng, metric, alpha=0.3)
                plt.ylabel('value')
                plt.legend(loc='upper left')
                col += 1
            if combine:
                spg.set_ylim(np.min(m_min), np.max(m_max))
            if i == 0:
                plt.title(title)
            i += rows
        out_file = f"output/{name}.png"
        plt.savefig(out_file)
        print(f"SAVED {out_file}   (run \033[94mpython serve.py\033[00m to access webserver on port 8080)")

        ## save models
        if save_model:
            for net in model.layers:
                model_file = model.model_files[net.name]
                net.save_weights(model_file)
                print(f"SAVED {net.name} weights to {model_file}")


class GeneralAI(tf.keras.Model):
    """General AI"""
    def __init__(self, arch, env, trader, env_render, save_model, chkpts, max_episodes, max_steps, learn_rates,
                 latent_size, latent_dist, mixture_multi, net_lstm, net_attn, aio_max_latents, aug_data_step, aug_data_pos):
        super().__init__()
        compute_dtype = tf.dtypes.as_dtype(self.compute_dtype)
        self.float_max = tf.constant(compute_dtype.max, compute_dtype)
        self.float_maxroot = tf.constant(tf.math.sqrt(self.float_max), compute_dtype)
        self.float_eps = tf.constant(tf.experimental.numpy.finfo(compute_dtype).eps, compute_dtype)
        self.float_eps_max = tf.constant(1.0 / self.float_eps, compute_dtype)
        self.loss_scale = tf.math.exp(tf.math.log(self.float_eps_max) * (1/2))
        self.float64_zero = tf.constant(0, tf.float64)

        self.arch, self.env, self.trader, self.env_render, self.save_model = arch, env, trader, env_render, save_model
        self.chkpts = tf.constant(chkpts, tf.int32)
        self.max_episodes = tf.constant(max_episodes, tf.int32)
        self.max_steps = tf.constant(max_steps, tf.int32)
        self.learn_rates = {}
        for k, v in learn_rates.items():
            self.learn_rates[k] = tf.constant(v, compute_dtype)
        self.initializer = tf.keras.initializers.GlorotUniform(time.time_ns())

        self.obs_spec, self.obs_zero, _ = gym_util.get_spec(
            env.observation_space, space_name='obs', compute_dtype=self.compute_dtype,
            net_attn_io=net_attn['io'], aio_max_latents=aio_max_latents, mixture_multi=mixture_multi
        )
        self.action_spec, _, self.action_zero_out = gym_util.get_spec(
            env.action_space, space_name='actions',
            compute_dtype=self.compute_dtype, mixture_multi=mixture_multi
        )
        self.obs_spec_len, self.action_spec_len = len(self.obs_spec), len(self.action_spec)
        self.action_total_size = tf.constant(np.sum([np.prod(feat['step_shape']) for feat in self.action_spec]), compute_dtype)
        self.gym_step_shapes = [feat['step_shape'] for feat in self.obs_spec]
        self.gym_step_shapes += [tf.TensorShape((1, 1)), tf.TensorShape((1, 1)), tf.TensorShape((1, 2)) if trader else tf.TensorShape((1, 1))]
        self.gym_step_dtypes = [feat['dtype'] for feat in self.obs_spec] + [tf.float64, tf.bool, tf.float64]
        self.rewards_zero, self.dones_zero = tf.constant([[0]], tf.float64), tf.constant([[False]], tf.bool)
        self.step_zero, self.step_size_one = tf.constant([[0]]), tf.constant([[1]])

        latent_spec = {'dtype':compute_dtype, 'latent_size':latent_size, 'num_latents':1, 'max_latents':aio_max_latents, 'max_batch_out':1}
        latent_spec.update({'inp':512, 'midp':256, 'outp':512, 'evo':64})

        if latent_dist == 'd': # deterministic
            latent_spec.update({'dist_type':'d', 'num_components':latent_size, 'event_shape':(latent_size,)})
        if latent_dist == 'c': # categorical
             # TODO https://keras.io/examples/generative/vq_vae/
            latent_spec.update({'dist_type':'c', 'num_components':0, 'event_shape':(latent_size, latent_size)})
        if latent_dist == 'mx': # continuous
            latent_spec.update({'dist_type':'mx', 'num_components':int(latent_size/16), 'event_shape':(latent_size,)})

        if aug_data_step:
            self.obs_spec += [{
                'space_name':'step', 'name':'', 'event_shape':(1,), 'event_size':1,
                'channels':1, 'step_shape':tf.TensorShape((1, 1)), 'num_latents':1
            }]
        self.obs_spec += [{
            'space_name':'reward_prev', 'name':'', 'event_shape':(1,), 'event_size':1,
            'channels':1, 'step_shape':tf.TensorShape((1, 1)), 'num_latents':1
        }]
        inputs = {'obs':self.obs_zero, 'step':[self.step_zero], 'reward_prev':[self.rewards_zero], 'return_goal':[self.rewards_zero]}

        if arch in ('PG',):
            opt_spec = [{'name':'action', 'type':'a', 'schedule_type':'', 'learn_rate':self.learn_rates['action'], 'float_eps':self.float_eps}]
            stats_spec = [
                {'name':'rwd', 'b1':0.99, 'b2':0.99, 'dtype':tf.float64},
                {'name':'loss', 'b1':0.99, 'b2':0.99, 'dtype':compute_dtype},
                {'name':'delta', 'b1':0.99, 'b2':0.99, 'dtype':compute_dtype}
            ]
            self.action = nets.ArchFull(
                'A', inputs, opt_spec, stats_spec, self.obs_spec, self.action_spec, latent_spec, obs_latent=False,
                net_blocks=2, net_lstm=net_lstm, net_attn=net_attn, num_heads=4, memory_size=max_steps, aug_data_pos=aug_data_pos
            )
            outputs = self.action(inputs)
            self.action.optimizer_weights = util.optimizer_build(self.action.optimizer['action'], self.action.trainable_variables)
            util.net_build(self.action, self.initializer)

        self.metrics_spec()
        self.reset_states = tf.function(self.reset_states, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
        self.reset_states()
        arch_run = getattr(self, arch)
        arch_run = tf.function(arch_run, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
        setattr(self, arch, arch_run)

    def metrics_spec(self):
        metrics_loss = OrderedDict()
        metrics_loss['2rewards*'] = {'-rewards_ma':np.float64, '-rewards_total+':np.float64, 'rewards_final=':np.float64}
        metrics_loss['1steps'] = {'steps+':np.int64}
        if self.arch == 'PG':
            metrics_loss['1nets*'] = {'-loss_ma':np.float64, '-loss_action':np.float64}
            metrics_loss['1extras'] = {'loss_action_returns':np.float64}
            metrics_loss['1extras2*'] = {'actlog0':np.float64, 'actlog1':np.float64}

        for loss_group in metrics_loss.values():
            for k in loss_group.keys():
                if k.endswith('=') or k.endswith('+'):
                    loss_group[k] = [0 for i in range(self.max_episodes)]
                else:
                    loss_group[k] = [[] for i in range(self.max_episodes)]
        self.metrics_loss = metrics_loss

    def metrics_update(self, *args):
        args = list(args)
        log_metrics, episode, idx = args[0], args[1], 2
        for loss_group in self.metrics_loss.values():
            for k in loss_group.keys():
                if log_metrics[idx-2]:
                    if k.endswith('='):
                        loss_group[k][episode] = args[idx]
                    elif k.endswith('+'):
                        loss_group[k][episode] += args[idx]
                    else:
                        loss_group[k][episode] += [args[idx]]
                idx += 1
        return np.asarray(0, np.int32) # dummy

    def env_reset(self, dummy):
        obs, reward, done, metrics = self.env.reset(), 0.0, False, [0]
        if self.env_render:
            self.env.render()
        if hasattr(self.env, 'np_struc'):
            rtn = gym_util.struc_to_feat(obs[0])
            metrics = obs[1]['metrics']
        else:
            rtn = gym_util.space_to_feat(obs, self.env.observation_space)
        rtn += [np.asarray([[reward]], np.float64), np.asarray([[done]], bool), np.asarray([metrics], np.float64)]
        return rtn

    def env_step(self, *args): # args = tuple of ndarrays
        if hasattr(self.env, 'np_struc'):
            action = gym_util.out_to_struc(list(args), self.env.action_dtype)
        else:
            action = gym_util.out_to_space(args, self.env.action_space, [0])
        obs, reward, done, info = self.env.step(action)
        if self.env_render:
            self.env.render()
        if hasattr(self.env, 'np_struc'):
            rtn = gym_util.struc_to_feat(obs)
        else:
            rtn = gym_util.space_to_feat(obs, self.env.observation_space)
        metrics = info['metrics'] if 'metrics' in info else [0]
        rtn += [np.asarray([[reward]], np.float64), np.asarray([[done]], bool), np.asarray([metrics], np.float64)]
        return rtn

    def checkpoints(self, *args):
        model_files = ""
        for layer in self.layers:
            model_file = self.model_files[layer.name]
            layer.save_weights(model_file)
            model_files += ' ' + model_file.split('/')[-1]
        print(f"SAVED {model_files}")
        return np.asarray(0, np.int32) # dummy

    def reset_states(self, use_img=False):
        """https://www.tensorflow.org/api_docs/python/tf/keras/Model#reset_states"""
        for layer in self.layers:
            if hasattr(layer, 'reset_states'):
                layer.reset_states(use_img=use_img)

    def PG_actor(self, inputs, return_goal):
        print("tracing -> GeneralAI PG_actor")
        obs, actions = [None]*self.obs_spec_len, [None]*self.action_spec_len
        for i in range(self.obs_spec_len):
            obs[i] = tf.TensorArray(self.obs_spec[i]['dtype'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.obs_spec[i]['step_shape'][1:])
        for i in range(self.action_spec_len):
            actions[i] = tf.TensorArray(
                self.action_spec[i]['dtype_out'], size=1, dynamic_size=True,
                infer_shape=False, element_shape=self.action_spec[i]['step_shape'][1:]
            )
        rewards = tf.TensorArray(tf.float64, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        dones = tf.TensorArray(tf.bool, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        returns = tf.TensorArray(tf.float64, size=0, dynamic_size=True, infer_shape=False, element_shape=(1,))

        step = tf.constant(0)
        while not inputs['dones'][-1][0]:
            for i in range(self.obs_spec_len):
                obs[i] = obs[i].write(step, inputs['obs'][i][-1])

            action = [None]*self.action_spec_len
            inputs_step = {'obs':inputs['obs'], 'step':[tf.reshape(step, (1, 1))], 'reward_prev':[inputs['rewards']], 'return_goal':[return_goal]}
            action_logits = self.action(inputs_step)
            for i in range(self.action_spec_len):
                action_dist = self.action.dist[i](action_logits[i])
                action[i] = action_dist.sample()

            action_dis = [None]*self.action_spec_len
            for i in range(self.action_spec_len):
                actions[i] = actions[i].write(step, action[i][0])
                action_dis[i] = util.discretize(action[i][0], self.action_spec[i])

            np_in = tf.numpy_function(self.env_step, action_dis, self.gym_step_dtypes)
            for i in range(len(np_in)):
                np_in[i].set_shape(self.gym_step_shapes[i])
            inputs['obs'], inputs['rewards'], inputs['dones'] = np_in[:-3], np_in[-3], np_in[-2]

            rewards = rewards.write(step, inputs['rewards'][-1])
            dones = dones.write(step, inputs['dones'][-1])
            returns = returns.write(step, [self.float64_zero])
            returns_updt = returns.stack()
            returns_updt = returns_updt + inputs['rewards'][-1]
            returns = returns.unstack(returns_updt)
            step += 1

        outputs = {}
        out_obs, out_actions = [None]*self.obs_spec_len, [None]*self.action_spec_len
        for i in range(self.obs_spec_len):
            out_obs[i] = obs[i].stack()
        for i in range(self.action_spec_len):
            out_actions[i] = actions[i].stack()
        outputs['obs'], outputs['actions'] = out_obs, out_actions
        outputs['rewards'], outputs['dones'], outputs['returns'] = rewards.stack(), dones.stack(), returns.stack()
        return outputs, inputs

    def PG_learner_onestep(self, inputs, training=True):
        print("tracing -> GeneralAI PG_learner_onestep")
        loss = {}
        loss_actions_lik = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_actions = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        metric_actlog = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(2,))

        inputs_rewards = tf.concat([self.rewards_zero, inputs['rewards']], axis=0)
        returns = inputs['returns'][0:1]
        for step in tf.range(tf.shape(inputs['dones'])[0]):
            obs = [None]*self.obs_spec_len
            for i in range(self.obs_spec_len):
                obs[i] = inputs['obs'][i][step:step+1]
                obs[i].set_shape(self.obs_spec[i]['step_shape'])
            action = [None]*self.action_spec_len
            for i in range(self.action_spec_len):
                action[i] = inputs['actions'][i][step:step+1]
                action[i].set_shape(self.action_spec[i]['step_shape'])
            returns_calc = tf.squeeze(tf.cast(returns, self.compute_dtype), axis=-1)

            inputs_step = {'obs':obs, 'step':[tf.reshape(step, (1, 1))], 'reward_prev':[inputs_rewards[step:step+1]], 'return_goal':[returns]}
            with tf.GradientTape() as tape_action:
                action_logits = self.action(inputs_step)
                action_dist = [None]*self.action_spec_len
                for i in range(self.action_spec_len):
                    action_dist[i] = self.action.dist[i](action_logits[i])
                loss_action_lik = util.loss_likelihood(action_dist, action)
                loss_action = loss_action_lik * returns_calc
            if loss_action_lik > self.float_eps:
                gradients = tape_action.gradient(loss_action, self.action.trainable_variables)
                self.action.optimizer['action'].apply_gradients(zip(gradients, self.action.trainable_variables))
            loss_actions_lik = loss_actions_lik.write(step, loss_action_lik / self.action_total_size)
            loss_actions = loss_actions.write(step, loss_action)
            metric_actlog = metric_actlog.write(step, action_logits[0][0][0:2])

        loss['action_lik'], loss['action'], loss['actlog'] = loss_actions_lik.concat(), loss_actions.concat(), metric_actlog.stack()
        return loss

    def PG(self):
        """Policy Gradient: https://www.youtube.com/watch?v=A_2U6Sx67sE """
        print("tracing -> GeneralAI PG")
        tf.print("RUNNING")
        return_goal = tf.constant([[-self.loss_scale.numpy()]], tf.float64)
        ma, ma_loss, snr_loss = tf.constant(0, tf.float64), self.float_maxroot, tf.constant(1, self.compute_dtype)
        episode = tf.constant(0)
        while episode < self.max_episodes:
            tf.autograph.experimental.set_loop_options(parallel_iterations=1)
            np_in = tf.numpy_function(self.env_reset, [tf.constant(0)], self.gym_step_dtypes)
            for i in range(len(np_in)):
                np_in[i].set_shape(self.gym_step_shapes[i])
            inputs = {'obs':np_in[:-3], 'rewards':np_in[-3], 'dones':np_in[-2]}

            self.reset_states()
            outputs, inputs = self.PG_actor(inputs, return_goal)
            rewards_total = outputs['returns'][0][0]
            util.stats_update(self.action.stats['rwd'], rewards_total)
            ma, ema, snr, std = util.stats_get(self.action.stats['rwd'])

            self.reset_states()
            loss = self.PG_learner_onestep(outputs)
            util.stats_update(self.action.stats['loss'], tf.math.reduce_mean(loss['action_lik']))
            ma_loss, ema_loss, snr_loss, std_loss = util.stats_get(self.action.stats['loss'])
            self.action.optimizer['action'].learning_rate = self.learn_rates['action'] * snr_loss

            log_metrics = [True, True, True, True, True, True, True, True, True, True, True, True, True, True]
            metrics = [
                log_metrics, episode, ma, tf.math.reduce_sum(outputs['rewards']),
                outputs['rewards'][-1][0], tf.shape(outputs['rewards'])[0],
                ma_loss, tf.math.reduce_mean(loss['action_lik']),
                tf.math.reduce_mean(loss['action']),
                tf.math.reduce_mean(loss['actlog'][:, 0]), tf.math.reduce_mean(loss['actlog'][:, 1]),
            ]
            dummy = tf.numpy_function(self.metrics_update, metrics, [tf.int32])

            if self.save_model:
                if episode > tf.constant(0) and episode % self.chkpts == tf.constant(0):
                    tf.numpy_function(self.checkpoints, [tf.constant(0)], [tf.int32])
            episode += 1


if __name__ == '__main__':
    main()
