import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_discrete import PPO_discrete
from env import Env
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


def evaluate_policy(args, env, agent, state_norm):
    s = env.reset()
    invalid_choice = env.get_invalid_action()
    if args.use_state_norm:  # During the evaluating,update=False
        s = state_norm(s, update=False)
    done = False
    while not done:
        a = agent.evaluate(s,invalid_choice)  # We use the deterministic policy during the evaluating
        s_, r, done, total_revenue,travel_cost,cruise_cost = env.step(a)
        invalid_choice = env.get_invalid_action()
        if args.use_state_norm:
            s_ = state_norm(s_, update=False)
        s = s_

    return env.accumulative_rewards


def main(args,env_name,number):

    env = Env()
    env_evaluate = Env(evaluate=True)  # When evaluating the policy, we need to rebuild an environment

    args.state_dim = env.observation_space
    args.action_dim = env.action_space
    args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    print("env={}".format(env_name))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_episode_steps={}".format(args.max_episode_steps))

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training
    episode_steps = 0  # 记录episode长度

    replay_buffer = ReplayBuffer(args)
    agent = PPO_discrete(args)

    # Build a tensorboard
    writer = SummaryWriter(log_dir='runs/PPO_discrete/env_{}_number_{}'.format(env_name, number))

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    while total_steps < args.max_train_steps:
        s = env.reset()
        invalid_choice = env.get_invalid_action()
        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()
        done = False
        while not done:
            a, a_logprob = agent.choose_action(s,invalid_choice)  # Action and the corresponding log probability
            s_, r, done,revenue,travel_t,cruise_t = env.step(a)
            invalid_choice = env.get_invalid_action()
            if not done:
                if args.use_state_norm:
                    s_ = state_norm(s_)
                if args.use_reward_norm:
                    r = reward_norm(r)
                elif args.use_reward_scaling:
                    r = reward_scaling(r)

            replay_buffer.store(s, a, a_logprob, r, s_,done)
            s = s_
            total_steps += 1

            # When the number of transitions in buffer reaches batch_size,then update
            # 网络更新频率  每个batch更新一次
            if replay_buffer.count == args.batch_size:
                cri_loss,entropy = agent.update(replay_buffer,total_steps)
                replay_buffer.count = 0

                writer.add_scalar('critic_loss', cri_loss, global_step=total_steps)
                writer.add_scalar('dist_entropy', entropy, global_step=total_steps)

        episode_steps += 1
        writer.add_scalar('train_episode_reward', env.accumulative_rewards, global_step=total_steps)
        print(f"episode: {episode_steps}")
        print(
            f"total rev:{env.total_revenue},park rev:{env.park_revenue},char rev:{env.char_revenue},park refuse:{env.park_refuse},char refuse:{env.char_refuse}")
        print(f"travel cost:{env.travel_cost},cruise cost:{env.cruise_cost}")

        # Evaluate the policy every 'evaluate_freq' steps
        if episode_steps % args.evaluate_freq == 0:
            evaluate_num += 1
            evaluate_reward = evaluate_policy(args, env_evaluate, agent, state_norm)
            evaluate_rewards.append(evaluate_reward)
            print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
            print(f"total rev:{env_evaluate.total_revenue},park rev:{env_evaluate.park_revenue},char rev:{env_evaluate.char_revenue},park refuse:{env_evaluate.park_refuse},char refuse:{env_evaluate.char_refuse}")
            print(f"travel cost:{env_evaluate.travel_cost},cruise cost:{env_evaluate.cruise_cost}")
            writer.add_scalar('evaluate_episode_rewards', evaluate_rewards[-1], global_step=total_steps)
            # Save the rewards
            if evaluate_num % args.save_freq == 0:
                np.save('./data_train/PPO_discrete_env_{}_number_{}.npy'.format(env_name, number), np.array(evaluate_rewards))




if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=int, default=3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=5, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=128, help="The number of neurons in hidden layers of the neural network")
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

    env_name = 'online_assign'
    main(args, env_name=env_name, number=7)
