import torch
import numpy as np
import copy
import torch.nn.functional as F
import torch.autograd as autograd
from network import Dueling_Net, Net, ICM

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args,
                                                                                                                **kwargs)


class DQN(object):
    def __init__(self, args):
        self.action_dim = args.action_dim
        self.batch_size = args.batch_size  # batch size
        self.max_train_steps = args.max_train_steps
        self.lr = args.lr  # learning rate
        self.gamma = args.gamma  # discount factor
        self.tau = args.tau  # Soft update
        self.use_soft_update = args.use_soft_update
        self.target_update_freq = args.target_update_freq  # hard update
        self.update_count = 0

        self.grad_clip = args.grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_double = args.use_double
        self.use_dueling = args.use_dueling
        self.use_per = args.use_per
        self.use_n_steps = args.use_n_steps
        self.use_icm = args.use_icm
        if self.use_n_steps:
            self.gamma = self.gamma ** args.n_steps

        if self.use_dueling:  # Whether to use the 'dueling network'
            self.net = Dueling_Net(args)
        else:
            self.net = Net(args)

        self.target_net = copy.deepcopy(self.net)  # Copy the online_net to the target_net

        self.USE_CUDA = USE_CUDA

        if self.use_icm:
            # icm related
            self.forward_scale = args.forward_scale  # 0.8
            self.inverse_scale = args.inverse_scale  # 0.2
            self.intrinsic_scale = args.intrinsic_scale  # 1
            self.use_extrinsic = args.use_extrinsic
            self.ICM = ICM(args)

            if USE_CUDA:
                self.net = self.net.cuda()
                self.target_net = self.target_net.cuda()
                self.ICM = self.ICM.cuda()

            self.optimizer = torch.optim.Adam(list(self.net.parameters()) + list(self.ICM.parameters()), lr=self.lr)
        else:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def choose_action(self, state, epsilon, invalid_action=None):
        with torch.no_grad():
            state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0)
            if USE_CUDA:
                state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0).cuda()
            q = self.net(state)

            if invalid_action is not None:
                q = q + (invalid_action - 1) * 1e6

            if np.random.uniform() > epsilon:
                action = q.argmax(dim=-1).item()
            else:
                # action = np.random.randint(0, self.action_dim)
                action = np.random.choice(np.where(np.array(invalid_action) == 1)[0].tolist())
            return action

    def learn(self, replay_buffer, total_steps):
        batch, batch_index, IS_weight = replay_buffer.sample(total_steps)

        with torch.no_grad():  # q_target has no gradient
            if self.use_double:  # Whether to use the 'double q-learning'
                # Use online_net to select the action
                next_q_values = self.net(batch['next_state'])
                next_q_values = next_q_values + (batch['next_invalid_action'] - 1) * 1e6
                a_argmax = next_q_values.argmax(dim=-1, keepdim=True)  # shape：(batch_size,1)
                # a_argmax = self.net(batch['next_state']).argmax(dim=-1, keepdim=True)  # shape：(batch_size,1)

                # Use target_net to estimate the q_target
                q_target = batch['reward'] + self.gamma * (1 - batch['done']) * self.target_net(
                    batch['next_state']).gather(-1, a_argmax).squeeze(-1)  # shape：(batch_size,)
            else:
                q_target = batch['reward'] + self.gamma * (1 - batch['done']) * \
                           self.target_net(batch['next_state']).max(dim=-1)[0]  # shape：(batch_size,)

        q_current = self.net(batch['state']).gather(-1, batch['action']).squeeze(-1)  # shape：(batch_size,)
        td_errors = q_current - q_target  # shape：(batch_size,)

        if self.use_per:
            loss = (IS_weight * (td_errors ** 2)).mean()
            replay_buffer.update_batch_priorities(batch_index, td_errors.detach().numpy())
        else:
            loss = (td_errors ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.optimizer.step()

        if self.use_soft_update:  # soft update
            for param, target_param in zip(self.net.parameters(), self.target_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:  # hard update
            self.update_count += 1
            if self.update_count % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.net.state_dict())

        if self.use_lr_decay:  # learning rate Decay
            self.lr_decay(total_steps)

        return loss.item()

    def learn_icm(self, replay_buffer, total_steps):
        batch, batch_index, IS_weight = replay_buffer.sample(total_steps)

        if USE_CUDA:
            batch['action'] = batch['action'].cuda()
            batch['reward'] = batch['reward'].cuda()
            batch['done'] = batch['done'].cuda()

        a_vec = F.one_hot(batch['action'], num_classes=self.action_dim).view(-1, self.action_dim)
        pred_s_, pred_a_vec, f_s_ = self.ICM.get_full(batch['state'], batch['next_state'], a_vec)
        forward_loss = F.mse_loss(pred_s_, f_s_.detach(), reduction='none')
        inverse_pred_loss = F.cross_entropy(pred_a_vec, a_vec.float().detach(), reduction='none')

        intrinsic_rewards = self.intrinsic_scale * forward_loss.mean(-1)
        total_rewards = intrinsic_rewards.clone()
        total_rewards = (total_rewards - min(total_rewards)) / (max(total_rewards) - min(total_rewards))
        total_rewards = 0.003 * total_rewards

        if self.use_extrinsic:
            total_rewards += batch['reward']

        with torch.no_grad():  # q_target has no gradient
            if self.use_double:  # Whether to use the 'double q-learning'
                # Use online_net to select the action
                next_q_values = self.net(batch['next_state'])
                next_q_values = next_q_values + (batch['next_invalid_action'] - 1) * 1e6
                a_argmax = next_q_values.argmax(dim=-1, keepdim=True)  # shape：(batch_size,1)
                # a_argmax = self.net(batch['next_state']).argmax(dim=-1, keepdim=True)  # shape：(batch_size,1)
                q_target = total_rewards + self.gamma * (1 - batch['done']) * self.target_net(
                    batch['next_state']).gather(-1, a_argmax).squeeze(-1)
                # Use target_net to estimate the q_target

            else:
                q_target = total_rewards + self.gamma * (1 - batch['done']) * \
                           self.target_net(batch['next_state']).max(dim=-1)[0]

        q_current = self.net(batch['state']).gather(-1, batch['action']).squeeze(-1)  # shape：(batch_size,)
        td_errors = q_current - q_target  # shape：(batch_size,)

        if self.use_per:
            Q_loss = (IS_weight * (td_errors ** 2)).mean()
            replay_buffer.update_batch_priorities(batch_index, td_errors.detach().numpy())
            loss = Q_loss + self.forward_scale * forward_loss.mean() + self.inverse_scale * inverse_pred_loss.mean()

        else:
            Q_loss = (td_errors ** 2).mean()
            loss = Q_loss + self.forward_scale * forward_loss.mean() + self.inverse_scale * inverse_pred_loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.optimizer.step()

        if self.use_soft_update:  # soft update
            for param, target_param in zip(self.net.parameters(), self.target_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:  # hard update
            self.update_count += 1
            if self.update_count % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.net.state_dict())

        if self.use_lr_decay:  # learning rate Decay
            self.lr_decay(total_steps)

        return loss, Q_loss.item(), forward_loss.mean().item(), inverse_pred_loss.mean().item(), intrinsic_rewards.mean().item(),total_rewards

    def lr_decay(self, total_steps):
        lr_now = 0.9 * self.lr * (1 - total_steps / self.max_train_steps) + 0.1 * self.lr
        for p in self.optimizer.param_groups:
            p['lr'] = lr_now
