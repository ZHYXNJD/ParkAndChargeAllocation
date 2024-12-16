import json
import os

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_discrete import PPO_discrete
from DQN import env_0817
# from new_env import Env
import warnings


warnings.filterwarnings('ignore', category=UserWarning)


def write_args(f, args):
    attributes = {}
    for key, value in args.__dict__.items():
        attributes[key] = str(value)
    f.write(json.dumps(attributes))


def save_data(number, seed, episode_data, train_loss, step_reward):
    try:
        os.makedirs('../save_data_reinforce/save_ppo/data_train/number_{}_seed_{}'.format(number, seed))
    except:
        pass
    pd.DataFrame(columns={'step', 'loss'}, data=train_loss).to_csv(
        '../save_data_reinforce/save_ppo/data_train/number_{}_seed_{}/train_loss.csv'.format(number, seed))

    column_name = ['episode', 'acc rewards', 'total rev', 'park rev', 'char rev', 'park refuse', 'char refuse',
                   'travel cost', 'cruise cost']
    pd.DataFrame(columns=column_name, data=episode_data).to_csv(
        '../save_data_reinforce/save_ppo/data_train/number_{}_seed_{}/episode_data.csv'.format(number, seed))
    # pd.DataFrame(columns={'step', 'reward'}, data=step_reward).to_csv(
    #     '../save_data_reinforce/save_ppo/data_train/number_{}_seed_{}/step_reward.csv'.format(number, seed))


def save_episode_assign_data(number, seed, episode_number, assign_info):
    try:
        os.makedirs('../save_data_reinforce/save_ppo/data_episode/number_{}_seed_{}'.format(number, seed))
    except:
        pass
    pd.DataFrame(columns={'t', 'req_id', 'req_info', 'pl_num'}, data=assign_info).to_csv(
        '../save_data_reinforce/save_ppo/data_episode/number_{}_seed_{}/episode_{}_assign.csv'.format(number, seed,
                                                                                                      episode_number))


def main(args, number,seed):
    env = env_0817.Env(park_arrival_num=args.park_arrival_num, charge_ratio=args.charge_ratio, rule=args.rule)

    args.state_dim = env.observation_space
    args.action_dim = env.action_space
    env.acc = args.acc
    env.r_c_no_avail = args.r_c_no_avail
    env.r_p_no_avail = args.r_p_no_avail
    print("park_num={}".format(args.park_arrival_num))
    print("charge_ratio={}".format(args.charge_ratio))
    print("match_rule={}".format(args.rule))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))

    total_steps = 0  # Record the total steps during the training

    replay_buffer = ReplayBuffer(args)
    agent = PPO_discrete(args)

    # Build a tensorboard
    writer = SummaryWriter(log_dir='../save_data_reinforce/save_ppo/runs/number_{}_seed_{}'.format(number, seed))

    with open('../save_data_reinforce/save_ppo/configs/number_{}_seed_{}.txt'.format(number, seed),
              'w') as f:
        write_args(f, args)

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    episode_result = []  # 存放episode data
    train_loss = []
    step_reward = []
    episode_steps = 0

    while total_steps < args.max_train_steps:
        s = env.reset()
        this_mask = env.get_invalid_action()
        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()

        done = False
        train_reward = 0
        assign_info = []

        while not done:

            a, a_logprob = agent.choose_action(s, this_mask)  # Action and the corresponding log probability

            # assign information
            if (episode_steps + 1) % 10 == 0:
                assign_info.append([env.t, env.req_id_at_t, env.req_t, a])

            s_, r, done = env.step(a)
            next_mask = env.get_invalid_action()

            if args.use_state_norm:
                s_ = state_norm(s_)
            if args.use_reward_norm:
                r = reward_norm(r)
            elif args.use_reward_scaling:
                r = reward_scaling(r)

            train_reward += r
            total_steps += 1
            step_reward.append([total_steps, r])

            replay_buffer.store(s, a, a_logprob, r, s_, done, this_mask, next_mask)
            s = s_
            this_mask = env.get_invalid_action()

            # When the number of transitions in buffer reaches batch_size,then update
            # 网络更新频率  每个batch更新一次
            if replay_buffer.count == args.batch_size:
                cri_loss, entropy = agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

                writer.add_scalar('critic_loss', cri_loss, global_step=episode_steps)
                writer.add_scalar('dist_entropy', entropy, global_step=episode_steps)
                train_loss.append([total_steps, cri_loss])


        episode_steps += 1
        writer.add_scalar('episode_reward', train_reward, global_step=episode_steps)

        if episode_steps % 10 == 0:
            save_episode_assign_data(number=number, seed=seed, episode_number=episode_steps,
                                          assign_info=assign_info)

        print(f"episode: {episode_steps}, episode reward: {train_reward}")
        print(
            f"total rev:{env.total_revenue},park rev:{env.park_revenue},char rev:{env.char_revenue},park refuse:{env.park_refuse},char refuse:{env.char_refuse}")
        print(f"travel cost:{env.travel_cost},cruise cost:{env.cruise_cost}")
        print("\n")

        episode_result.append([episode_steps, env.accumulative_rewards, env.total_revenue,
                               env.park_revenue, env.char_revenue, env.park_refuse,
                               env.char_refuse, env.travel_cost, env.cruise_cost])

    save_data(number=number, seed=seed, episode_data=episode_result, train_loss=train_loss,
                   step_reward=step_reward)

    #  save model
    torch.save(agent.actor.state_dict(),
               f'../save_data_reinforce/save_ppo/model/number_{number}_seed_{seed}_net.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")

    parser.add_argument("--park_arrival_num", type=int, default=225, help="park arrival number")
    parser.add_argument("--charge_ratio", type=float, default=1, help="charge ratio")
    parser.add_argument("--rule", type=int, default=2, help="match rule")

    parser.add_argument("--objective:max revenue", type=int, default=0)
    parser.add_argument("--acc", type=int, default=1)
    parser.add_argument("--r_with_no_d", type=int, default=1)
    parser.add_argument("--r_p_no_avail", type=int, default=-5)
    parser.add_argument("--r_c_no_avail", type=int, default=-10)

    parser.add_argument("--max_train_steps", type=int, default=2000 * 500, help=" Maximum number of training steps")
    # parser.add_argument("--evaluate_freq", type=int, default=100, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=5, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=512, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=256,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=1e-5, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=1e-5, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=20, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=False, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    args = parser.parse_args()

    # env_name = 'Assign'

    main(args, number=9,seed=100)
