import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Dueling_Net(nn.Module):
    def __init__(self, args):
        super(Dueling_Net, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        if args.use_noisy:
            self.V = NoisyLinear(args.hidden_dim, 1)
            self.A = NoisyLinear(args.hidden_dim, args.action_dim)
        else:
            self.V = nn.Linear(args.hidden_dim, 1)
            self.A = nn.Linear(args.hidden_dim, args.action_dim)

    def forward(self, s):
        s = torch.relu(self.fc1(s))
        s = torch.relu(self.fc2(s))
        V = self.V(s)  # batch_size X 1
        A = self.A(s)  # batch_size X action_dim
        Q = V + (A - torch.mean(A, dim=-1, keepdim=True))  # Q(s,a)=V(s)+A(s,a)-mean(A(s,a))
        return Q


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        if args.use_noisy:
            self.fc3 = NoisyLinear(args.hidden_dim, args.action_dim)
        else:
            self.fc3 = nn.Linear(args.hidden_dim, args.action_dim)

    def forward(self, s):
        s = torch.relu(self.fc1(s))
        s = torch.relu(self.fc2(s))
        Q = self.fc3(s)
        return Q


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            self.reset_noise()
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)  # mul是对应元素相乘
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)

        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))  # 这里要除以out_features

    def reset_noise(self):
        epsilon_i = self.scale_noise(self.in_features)
        epsilon_j = self.scale_noise(self.out_features)
        self.weight_epsilon.copy_(torch.ger(epsilon_j, epsilon_i))
        self.bias_epsilon.copy_(epsilon_j)

    def scale_noise(self, size):
        x = torch.randn(size)  # torch.randn产生标准高斯分布
        x = x.sign().mul(x.abs().sqrt())
        return x


class ICM(nn.Module):
    def __init__(self, args):
        super(ICM, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.icm_dim)

        self.pred_module1 = nn.Linear(args.icm_dim + args.action_dim, args.hidden_dim)
        self.pred_module2 = nn.Linear(args.hidden_dim, args.icm_dim)

        self.invpred_molule1 = nn.Linear(args.icm_dim + args.icm_dim, args.hidden_dim)
        self.invpred_module2 = nn.Linear(args.hidden_dim, args.action_dim)

    def get_feature_s(self, s):
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s.view(s.size(0), -1)))
        return s

    def forward(self, s):
        feature_s = self.get_feature_s(s)
        return feature_s

    def get_full(self, s, s_, a):
        # get feature
        f_s = self.get_feature_s(s)
        f_s_ = self.get_feature_s(s_)

        pred_s_ = self.pred(f_s, a)  # pred next feature_state
        pred_a = self.invpred(f_s, f_s_)  # pred a

        return pred_s_, pred_a, f_s_

    def pred(self, f_s, a):
        pred_s_ = F.relu(self.pred_module1(torch.cat([f_s, a], dim=-1).detach()))
        pred_s_ = self.pred_module2(pred_s_)

        return pred_s_

    def invpred(self, f_s, f_s_):
        pred_a = F.relu(self.invpred_molule1(torch.cat([f_s, f_s_], dim=-1)))
        pred_a = self.invpred_module2(pred_a)
        pred_a = F.softmax(pred_a, dim=-1)

        return pred_a
