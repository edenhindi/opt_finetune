import numpy as np
import torch

from ppo_maxEnt.distributions import FixedCategorical
from torch import nn


def evaluate_procgen(actor_critic, eval_envs, env_name,
                     device, steps, logger, deterministic=True):
    rew_batch = []
    done_batch = []

    for t in range(steps):
        with torch.no_grad():
            _, action, _, dist_probs, eval_recurrent_hidden_states = actor_critic.act(
                logger.obs[env_name].float().to(device),
                logger.eval_recurrent_hidden_states[env_name],
                logger.eval_masks[env_name],
                deterministic=deterministic)

            # Observe reward and next obs
            next_obs, reward, done, infos = eval_envs.step(action.squeeze().cpu().numpy())
            logger.eval_masks[env_name] = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)
            logger.eval_recurrent_hidden_states[env_name] = eval_recurrent_hidden_states

            if 'env_reward' in infos[0]:
                rew_batch.append([info['env_reward'] for info in infos])
            else:
                rew_batch.append(reward)
            done_batch.append(done)


            logger.obs[env_name] = next_obs

    rew_batch = np.array(rew_batch)
    done_batch = np.array(done_batch)

    return rew_batch, done_batch


def maxEnt_oracle(obs_all, action):
    next_action = action.clone().detach()
    for i in range(len(action)):
        obs = obs_all[i].cpu().numpy()
        action_i = action[i]
        new_action_i = np.array([7])

        min_r = np.nonzero((obs[1] == 1))[0].min()
        max_r = np.nonzero((obs[1] == 1))[0].max()
        middle_r = int(min_r + (max_r - min_r + 1) / 2)

        min_c = np.nonzero((obs[1] == 1))[1].min()
        max_c = np.nonzero((obs[1] == 1))[1].max()
        middle_c = int(min_c + (max_c - min_c + 1) / 2)

        if action_i == 7:
            if (max_r + 1 < 64) and obs[0][max_r + 1, middle_c] == 0:
                new_action_i = np.array([3])
            elif (max_c + 1 < 64) and obs[0][middle_r, max_c + 1] == 0:
                new_action_i = np.array([7])
            elif (min_r - 1 > 0) and obs[0][min_r - 1, middle_c] == 0:
                new_action_i = np.array([5])
            else:
                new_action_i = np.array([1])
        elif action_i == 5:
            if (max_c + 1 < 64) and obs[0][middle_r, max_c + 1] == 0:
                new_action_i = np.array([7])
            elif (min_r - 1 > 0) and obs[0][min_r - 1, middle_c] == 0:
                new_action_i = np.array([5])
            elif (min_c - 1 > 0) and obs[0][middle_r, min_c - 1] == 0:
                new_action_i = np.array([1])
            else:
                new_action_i = np.array([3])
        elif action_i == 3:
            if (min_c - 1 > 0) and obs[0][middle_r, min_c - 1] == 0:
                new_action_i = np.array([1])
            elif (max_r + 1 < 64) and obs[0][max_r + 1, middle_c] == 0:
                new_action_i = np.array([3])
            elif (max_c + 1 < 64) and obs[0][middle_r, max_c + 1] == 0:
                new_action_i = np.array([7])
            else:
                new_action_i = np.array([5])
        elif action_i == 1:
            if (min_r - 1 > 0) and obs[0][min_r - 1, middle_c] == 0:
                new_action_i = np.array([5])
            elif (min_c - 1 > 0) and obs[0][middle_r, min_c - 1] == 0:
                new_action_i = np.array([1])
            elif (max_r + 1 < 64) and obs[0][max_r + 1, middle_c] == 0:
                new_action_i = np.array([3])
            else:
                new_action_i = np.array([7])

        next_action[i] = torch.tensor(new_action_i)

    return next_action



def evaluate_procgen_maxEnt_avepool_original_L2(actor_critic, eval_envs_dic, eval_envs_dic_full_obs, env_name,
                                             device, steps, logger, num_buffer, kernel_size=3, stride=3, deterministic=True, p_norm=2, neighbor_size=1):
    eval_envs = eval_envs_dic[env_name]
    eval_envs_full_obs = eval_envs_dic_full_obs[env_name]
    rew_batch = []
    int_rew_batch = []
    done_batch = []
    seed_batch = []
    down_sample_avg = nn.AvgPool2d(kernel_size, stride=stride)


    for t in range(steps):
        with torch.no_grad():
            _, action, _, dist_probs, eval_recurrent_hidden_states = actor_critic.act(
                logger.obs[env_name].float().to(device),
                logger.eval_recurrent_hidden_states[env_name],
                logger.eval_masks[env_name],
                deterministic=deterministic)


            # Observe reward and next obs
            next_obs, reward, done, infos = eval_envs.step(action.squeeze().cpu().numpy())
            next_obs_full, _, _, _ = eval_envs_full_obs.step(action.squeeze().cpu().numpy())

            logger.eval_masks[env_name] = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)
            logger.eval_recurrent_hidden_states[env_name] = eval_recurrent_hidden_states

            if t == 0:
                prev_seeds = np.zeros_like(reward)
                for i in range(len(done)):
                    prev_seeds[i] = infos[i]['prev_level_seed']
                seed_batch.append(prev_seeds)

            seeds = np.zeros_like(reward)
            int_reward = np.zeros_like(reward)
            next_obs_ds = down_sample_avg(next_obs_full)
            for i in range(len(done)):
                seeds[i] = infos[i]['level_seed']
                if done[i] == 1 :
                    logger.obs_vec_ds[env_name][i] = []
                else:
                    env_steps = len(logger.obs_vec_ds[env_name][i])
                    if env_steps > 0:
                        if env_steps > num_buffer:
                            old_obs = torch.stack(logger.obs_vec_ds[env_name][i][env_steps-num_buffer:])
                        else:
                            old_obs = torch.stack(logger.obs_vec_ds[env_name][i])
                        neighbor_size_i = int(min(neighbor_size, len(logger.obs_vec_ds[env_name][i])) -1)
                        int_reward[i]  = (old_obs - next_obs_ds[i].unsqueeze(0)).flatten(start_dim=1).norm(p=p_norm, dim=1).sort().values[neighbor_size_i]

                logger.obs_vec_ds[env_name][i].append(next_obs_ds[i])


            rew_batch.append(reward)
            int_rew_batch.append(int_reward)
            done_batch.append(done)
            seed_batch.append(seeds)

            logger.obs[env_name] = next_obs
            logger.obs_full[env_name] = next_obs_full
            logger.last_action[env_name] = action

    rew_batch = np.array(rew_batch)
    int_rew_batch = np.array(int_rew_batch)
    done_batch = np.array(done_batch)
    seed_batch = np.array(seed_batch)

    return rew_batch, int_rew_batch, done_batch, seed_batch