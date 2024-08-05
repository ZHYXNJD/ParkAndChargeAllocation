import json
import os

from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from DQN.normalization import Normalization, RewardScaling
from replay_buffer import *
from rainbow_dqn import DQN
import argparse
# from new_env import Env
from nnew_env import Env


def write_args(f, args):
    attributes = {}
    for key, value in args.__dict__.items():
        attributes[key] = str(value)
    f.write(json.dumps(attributes))


class Runner:
    def __init__(self, args, number, seed):
        self.args = args
        self.number = number
        self.seed = seed

        self.env = Env(park_arrival_num=args.park_arrival_num, charge_ratio=args.charge_ratio, rule=args.rule)
        self.env.acc = args.acc
        self.env.r_with_no_d = args.r_with_no_d
        self.env.r_p_no_avail = args.r_p_no_avail
        self.env.r_c_no_avail = args.r_c_no_avail

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.args.state_dim = self.env.observation_space
        self.args.action_dim = self.env.action_space
        print("park_num={}".format(args.park_arrival_num))
        print("charge_ratio={}".format(args.charge_ratio))
        print("match_rule={}".format(args.rule))
        print("state_dim={}".format(args.state_dim))
        print("action_dim={}".format(args.action_dim))

        if args.use_per and args.use_n_steps:
            self.replay_buffer = N_Steps_Prioritized_ReplayBuffer(args)
        elif args.use_per:
            self.replay_buffer = Prioritized_ReplayBuffer(args)
        elif args.use_n_steps:
            self.replay_buffer = N_Steps_ReplayBuffer(args)
        else:
            self.replay_buffer = ReplayBuffer(args)
        self.agent = DQN(args)

        self.algorithm = 'save_DQN'
        if args.use_double and args.use_dueling and args.use_noisy and args.use_per and args.use_n_steps:
            self.algorithm = 'Rainbow_' + self.algorithm
        else:
            if args.use_double:
                self.algorithm += '_Double'
            if args.use_dueling:
                self.algorithm += '_Dueling'
            if args.use_noisy:
                self.algorithm += '_Noisy'
            if args.use_per:
                self.algorithm += '_PER'
            if args.use_n_steps:
                self.algorithm += "_N_steps"

        self.writer = SummaryWriter(log_dir='../save_data_reinforce/save_DQN/runs/{}_number_{}_seed_{}'.format(self.algorithm, number, seed))

        with open('../save_data_reinforce/save_DQN/configs/{}_number_{}_seed{}.txt'.format(self.algorithm, number, seed), 'w') as f:
            write_args(f, args)

        self.total_steps = 0  # Record the total steps during the training

        if args.use_noisy:  # 如果使用Noisy net，就不需要epsilon贪心策略了
            self.epsilon = 0
        else:
            self.epsilon = self.args.epsilon_init
            self.epsilon_min = self.args.epsilon_min
            self.epsilon_decay = (self.args.epsilon_init - self.args.epsilon_min) / self.args.epsilon_decay_steps

    def save_train_data(self,number,seed,episode_data,train_loss,step_reward):

        try:
            os.makedirs('../save_data_reinforce/save_DQN/data_train/number_{}_seed_{}'.format(number,seed))
        except:
            pass

        column_name = ['episode', 'acc rewards', 'total rev', 'park rev', 'char rev', 'park refuse', 'char refuse',
                       'travel cost', 'cruise cost']
        pd.DataFrame(columns=column_name,data=episode_data).to_csv('../save_data_reinforce/save_DQN/data_train/number_{}_seed_{}/episode_data.csv'.format(number,seed))
        pd.DataFrame(columns={'step','loss'},data=train_loss).to_csv('../save_data_reinforce/save_DQN/data_train/number_{}_seed_{}/train_loss.csv'.format(number,seed))
        pd.DataFrame(columns={'step','reward'},data=step_reward).to_csv('../save_data_reinforce/save_DQN/data_train/number_{}_seed_{}/step_reward.csv'.format(number,seed))

    def save_episode_assign_data(self,number,seed,episode_number,assign_info):
        try:
            os.makedirs('../save_data_reinforce/save_DQN/data_episode/number_{}_seed_{}'.format(number,seed))
        except:
            pass
        pd.DataFrame(columns={'t','req_id','req_info','pl_num'},data=assign_info).to_csv('../save_data_reinforce/save_DQN/data_episode/number_{}_seed_{}/episode_{}_assign.csv'.format(number,seed,episode_number))


    def run(self, ):
        state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
        if args.use_reward_norm:  # Trick 3:reward normalization
            reward_norm = Normalization(shape=1)
        elif args.use_reward_scaling:  # Trick 4:reward scaling
            reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

        episode_result = []  # 存放episode data
        train_loss = []
        step_reward = []
        episode_steps = 0
        while self.total_steps < self.args.max_train_steps:
            state = self.env.reset()
            this_invalid_choice = self.env.get_invalid_action()

            if args.use_state_norm:
                state = state_norm(state)
            if args.use_reward_scaling:
                reward_scaling.reset()

            done = False
            train_reward = 0
            assign_info = []

            """
            count13 
            count0
            countother
            """
            count13 = 0
            count0 = 0
            countother = 0
            while not done:
                action = self.agent.choose_action(state, epsilon=self.epsilon, invalid_action=this_invalid_choice)

                # print("episode:{},step:{},action:{},requset:{}".format(episode_steps, self.total_steps, action,
                #                                                        self.env.request_demand))

                # assign information
                if (episode_steps + 1) % 10 == 0:
                    assign_info.append([self.env.t, self.env.req_id_at_t, self.env.request_demand, action])

                next_state, reward, done = self.env.step(action)
                next_invalid_choice = self.env.get_invalid_action()

                if action == 13:
                    count13 += 1
                elif action == 0:
                    count0 += 1
                else:
                    countother += 1

                if not done:
                    if args.use_state_norm:
                        next_state = state_norm(next_state)
                    if args.use_reward_norm:
                        reward = reward_norm(reward)
                    elif args.use_reward_scaling:
                        reward = reward_scaling(reward)

                train_reward += reward
                self.total_steps += 1

                step_reward.append([self.total_steps,reward])

                if not self.args.use_noisy:  # Decay epsilon
                    self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon - self.epsilon_decay > self.epsilon_min else self.epsilon_min


                self.replay_buffer.store_transition(state, action, reward, next_state, this_invalid_choice,
                                                    next_invalid_choice, done)  # Store the transition
                state = next_state
                this_invalid_choice = next_invalid_choice

                loss = 0

                if self.replay_buffer.current_size >= self.args.batch_size:
                    if args.use_icm:
                        loss, Q_loss_record, forward_loss_record, inverse_pred_loss_record, intrinsic_rewards_rec = \
                            self.agent.learn_icm(self.replay_buffer, self.total_steps)

                        if self.total_steps % 500 == 0:
                            print(
                                "train_step: %5d,total_loss: %4f, forward_loss: %4f, inverse_pred_loss: %4f, Q_loss: %4f, intrinsic_rewards: %4f" %
                                (self.total_steps, loss, forward_loss_record, inverse_pred_loss_record,
                                 Q_loss_record, intrinsic_rewards_rec))
                    else:
                        loss = self.agent.learn(self.replay_buffer, self.total_steps)
                        train_loss.append([self.total_steps,loss])
                        if self.total_steps % 500 == 0:
                            print("train_step: %5d, Q_loss: %4f" % (self.total_steps, loss))
                # step reward
                self.writer.add_scalar('step_reward', reward, global_step=self.total_steps)
                # step loss
                self.writer.add_scalar('step_loss', loss, global_step=self.total_steps)

            print("episode{},count13:{},count0:{},count other:{}".format(episode_steps+1,count13,count0,countother))

            episode_steps += 1
            self.writer.add_scalar('episode_reward', train_reward, global_step=episode_steps)
            # if episode_steps % 10 == 0:
            if episode_steps % 10 == 0:
                self.save_episode_assign_data(number=self.number,seed=self.seed,episode_number=episode_steps,assign_info=assign_info)

            print(f"episode: {episode_steps}, episode reward: {train_reward}")
            print(
                f"total rev:{self.env.total_revenue},park rev:{self.env.park_revenue},char rev:{self.env.char_revenue},park refuse:{self.env.park_refuse},char refuse:{self.env.char_refuse}")
            print(f"travel cost:{self.env.travel_cost},cruise cost:{self.env.cruise_cost}")
            print("\n")

            episode_result.append([episode_steps, self.env.accumulative_rewards, self.env.total_revenue,
                                                     self.env.park_revenue, self.env.char_revenue, self.env.park_refuse,
                                                     self.env.char_refuse, self.env.travel_cost, self.env.cruise_cost])

        self.save_train_data(number=self.number,seed=self.seed,episode_data=episode_result,train_loss=train_loss,step_reward=step_reward)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for save_DQN")

    parser.add_argument("--park_arrival_num", type=int, default=400, help="park arrival number")
    parser.add_argument("--charge_ratio", type=float, default=0.25, help="charge ratio")
    parser.add_argument("--rule", type=int, default=2, help="match rule")

    parser.add_argument("--acc", type=int, default=10)
    parser.add_argument("--r_with_no_d", type=int, default=0)
    parser.add_argument("--r_p_no_avail", type=int, default=-10)
    parser.add_argument("--r_c_no_avail", type=int, default=-30)

    parser.add_argument("--max_train_steps", type=int, default=500 * 1440, help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=1e3,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")

    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")

    parser.add_argument("--buffer_capacity", type=int, default=int(1e4), help="The maximum replay-buffer capacity ")
    # parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--gamma", type=float, default=0.999, help="Discount factor")
    parser.add_argument("--epsilon_init", type=float, default=0.5, help="Initial epsilon")
    parser.add_argument("--epsilon_min", type=float, default=0.1, help="Minimum epsilon")
    parser.add_argument("--epsilon_decay_steps", type=int, default=int(1e5),
                        help="How many steps before the epsilon decays to the minimum")
    parser.add_argument("--tau", type=float, default=0.005, help="soft update the target network")
    parser.add_argument("--use_soft_update", type=bool, default=True, help="Whether to use soft update")
    parser.add_argument("--target_update_freq", type=int, default=200,
                        help="Update frequency of the target network(hard update)")
    parser.add_argument("--n_steps", type=int, default=5, help="n_steps")
    parser.add_argument("--alpha", type=float, default=0.6, help="PER parameter")
    parser.add_argument("--beta_init", type=float, default=0.4, help="Important sampling parameter in PER")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Learning rate Decay")
    parser.add_argument("--grad_clip", type=float, default=10.0, help="Gradient clip")

    parser.add_argument("--use_double", type=bool, default=True, help="Whether to use double Q-learning")
    parser.add_argument("--use_dueling", type=bool, default=True, help="Whether to use dueling network")
    parser.add_argument("--use_noisy", type=bool, default=False, help="Whether to use noisy network")
    parser.add_argument("--use_per", type=bool, default=True, help="Whether to use PER")
    parser.add_argument("--use_n_steps", type=bool, default=True, help="Whether to use n_steps Q-learning")

    parser.add_argument("--use_icm", type=bool, default=False, help="whether to use ICM module")
    parser.add_argument("--icm_dim", type=int, default=128, help="state feature dimension")
    # parser.add_argument("--forward_scale",type=float,default=0.8,help="used for predict next state feature")
    parser.add_argument("--forward_scale", type=float, default=0.1, help="used for predict next state feature")
    parser.add_argument("--inverse_scale", type=float, default=0.1, help="used for predict action")
    parser.add_argument("--intrinsic_scale", type=float, default=1, help="used for forward loss")
    parser.add_argument("--use_extrinsic", type=bool, default=False, help="whether use extrinsic reward")

    args = parser.parse_args()

    # for seed in [0, 10, 100]:
    #     runner = Runner(args=args,number=1, seed=seed)
    #     runner.run()
    runner = Runner(args=args, number=25, seed=1)
    runner.run()
