import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_discrete import PPO_discrete
# from env import Env
from new_env import Env
import warnings
from strategy.utils import save_episode_results

warnings.filterwarnings('ignore', category=UserWarning)


def evaluate_policy(args, env, agent, state_norm):
    s = env.reset()
    invalid_choice = env.get_invalid_action()
    if args.use_state_norm:  # During the evaluating,update=False
        s = state_norm(s, update=False)
    done = False
    result_list = []
    while not done:
        a = agent.evaluate(s, invalid_choice)  # We use the deterministic policy during the evaluating
        s_, r, done, result_list = env.step(a)
        invalid_choice = env.get_invalid_action()
        if args.use_state_norm:
            s_ = state_norm(s_, update=False)
        s = s_

    return env.accumulative_rewards, result_list


def main(args,env_name,number):
    env = Env(park_arrival_num=args.park_arrival_num, charge_ratio=args.charge_ratio, rule=args.rule)

    args.state_dim = env.observation_space
    args.action_dim = env.action_space
    args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    print("env={}".format(env_name))
    print("park_num={}".format(args.park_arrival_num))
    print("charge_ratio={}".format(args.charge_ratio))
    print("match_rule={}".format(args.rule))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_episode_steps={}".format(args.max_episode_steps))

    total_steps = 0  # Record the total steps during the training
    episode_steps = 0  # 记录episode长度

    replay_buffer = ReplayBuffer(args)
    agent = PPO_discrete(args)

    # Build a tensorboard
    writer = SummaryWriter(log_dir=r'G:\2023-纵向\停车分配\PPO\runs\{}_number_{}'.format(env_name,number))

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    column_name = ['episode', 'acc rewards', 'total rev', 'park rev', 'char rev', 'park refuse', 'char refuse',
                   'travel cost', 'cruise cost']
    episode_results = pd.DataFrame(columns=column_name)
    supply_matrix = []

    while total_steps < args.max_train_steps:
        s = env.reset()
        this_mask = env.get_invalid_action()
        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()
        done = False

        scaling_reward = 0

        while not done:
            a, a_logprob = agent.choose_action(s, this_mask)  # Action and the corresponding log probability
            s_, r, done, _ = env.step(a)
            next_mask = env.get_invalid_action()
            if not done:
                if args.use_state_norm:
                    s_ = state_norm(s_)
                if args.use_reward_norm:
                    r = reward_norm(r)
                elif args.use_reward_scaling:
                    r = reward_scaling(r)

            replay_buffer.store(s, a, a_logprob, r, s_, done,this_mask,next_mask)
            s = s_
            this_mask = env.get_invalid_action()
            total_steps += 1

            scaling_reward += r

            # When the number of transitions in buffer reaches batch_size,then update
            # 网络更新频率  每个batch更新一次
            if replay_buffer.count == args.batch_size:
                cri_loss, entropy = agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

                writer.add_scalar('critic_loss', cri_loss, global_step=episode_steps)
                writer.add_scalar('dist_entropy', entropy, global_step=episode_steps)

        episode_results.loc[len(episode_results)] = [episode_steps, env.accumulative_rewards, env.total_revenue,
                                                     env.park_revenue, env.char_revenue, env.park_refuse,
                                                     env.char_refuse, env.travel_cost, env.cruise_cost]
        supply_matrix.append(env.supply_t)
        episode_steps += 1
        writer.add_scalar('scaling_reward', scaling_reward, global_step=episode_steps)
        # writer.add_scalar('train_episode_reward', env.accumulative_rewards, global_step=episode_steps)
        print(f"episode: {episode_steps}")
        print(
            f"total rev:{env.total_revenue},park rev:{env.park_revenue},char rev:{env.char_revenue},park refuse:{env.park_refuse},char refuse:{env.char_refuse}")
        print(f"travel cost:{env.travel_cost},cruise cost:{env.cruise_cost}")

    # save results
    save_episode_results(episode_results, supply_matrix, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")

    parser.add_argument("--park_arrival_num", type=int, default=400, help="park arrival number")
    parser.add_argument("--charge_ratio", type=float, default=0.25, help="charge ratio")
    parser.add_argument("--rule", type=int, default=1, help="match rule")

    parser.add_argument("--max_train_steps", type=int, default=1500 * 1440, help=" Maximum number of training steps")
    # parser.add_argument("--evaluate_freq", type=int, default=100, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=5, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=256,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    args = parser.parse_args()

    # env_name = 'Assign'
    env_name = 'debug'
    main(args, env_name=env_name, number=9)
