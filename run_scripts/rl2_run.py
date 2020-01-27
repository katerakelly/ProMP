import os
import click
import json
import numpy as np

from maml_zoo.baselines.linear_baseline import LinearFeatureBaseline
from maml_zoo.envs.sawyer_envs.reacher.sawyer_reacher import SawyerReachingEnvMultitask, SawyerReachingEnvMultitaskVision
from maml_zoo.envs.rl2_env import rl2env
from maml_zoo.algos.vpg import VPG
from maml_zoo.algos.ppo import PPO
from maml_zoo.trainer import Trainer
from maml_zoo.samplers.maml_sampler import MAMLSampler
from maml_zoo.samplers.rl2_sample_processor import RL2SampleProcessor
from maml_zoo.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from maml_zoo.policies.gaussian_rnn_policy import GaussianRNNPolicy
from maml_zoo.logger import logger
maml_zoo_path = '/'.join(os.path.realpath(os.path.dirname(__file__)).split('/')[:-1])

def run_experiment(config):
    baseline = LinearFeatureBaseline() # KATE note this is no longer used!
    if config['obs_mode'] == 'image':
        env = rl2env(SawyerReachingEnvMultitaskVision())
    else:
        env = rl2env(SawyerReachingEnvMultitask())
    # obs is state, action, reward, and done
    obs_dim = np.prod(env.observation_space.shape) + np.prod(env.action_space.shape) + 1 + 1
    print('SCRIPT: obs dim', obs_dim)
    vision_args = None
    if config['obs_mode'] == 'image':
        vision_args = dict(base_depth=32, double_camera=config['double_camera'])

    policy = GaussianRNNPolicy(
            name="meta-policy",
            obs_dim=obs_dim,
            action_dim=np.prod(env.action_space.shape),
            meta_batch_size=config['meta_batch_size'],
            hidden_sizes=config['hidden_sizes'],
            cell_type=config['cell_type'],
            vision_args=vision_args
        )

    sampler = MAMLSampler(
        env=env,
        policy=policy,
        rollouts_per_meta_task=config['rollouts_per_meta_task'],  # This batch_size is confusing
        meta_batch_size=config['meta_batch_size'],
        max_path_length=config['max_path_length'],
        parallel=config['parallel'],
        envs_per_task=1,
    )

    sample_processor = RL2SampleProcessor(
        baseline=baseline,
        discount=config['discount'],
        gae_lambda=config['gae_lambda'],
        normalize_adv=config['normalize_adv'],
        positive_adv=config['positive_adv'],
    )

    algo = PPO(
        policy=policy,
        learning_rate=config['learning_rate'],
        max_epochs=config['max_epochs']
    )

    trainer = Trainer(
        algo=algo,
        policy=policy,
        env=env,
        sampler=sampler,
        sample_processor=sample_processor,
        n_itr=config['n_itr'],
    )
    trainer.train()

@click.command()
@click.argument('config', default='rl2_config.json')
@click.option('--log_dir', default='./data/rl2')
@click.option('--debug', is_flag=True, default=False)
def main(config, log_dir, debug):
    idx = np.random.randint(0, 1000)
    if debug:
        data_path = maml_zoo_path + '{}/debug'.format(log_dir)
    else:
        data_path = maml_zoo_path + '{}/test_{}'.format(log_dir, idx)
    logger.configure(dir=data_path, format_strs=['stdout', 'log', 'csv'],
                     snapshot_mode='last_gap')
    config = json.load(open(maml_zoo_path + "/configs/{}".format(config), 'r'))
    json.dump(config, open(data_path + '/params.json', 'w'))
    run_experiment(config)

if __name__=="__main__":
    main()
