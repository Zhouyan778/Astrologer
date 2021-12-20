import torch
from torch import nn
from torch.optim import Adam
import numpy as np
from optimization import BertAdam
from util import *
import geoopt.manifolds.poincare.math as pmath
import torch.nn.functional as F
import geoopt


def RMSE_error(pred, gold):
    return np.sqrt(np.mean((pred - gold) ** 2))


class Net(nn.Module):
    def __init__(self, config, device):
        super(Net, self).__init__()
        torch.manual_seed(config.SEED)
        torch.cuda.manual_seed(config.SEED)

        self.config = config
        self.n_class = config.event_class
        self.c = config.c
        self.embed = nn.Linear(in_features=self.n_class, out_features=1)

        self.emb_drop = nn.Dropout(p=config.dropout)

        self.gru1 = nn.GRUCell(input_size=self.n_class + self.config.emb_dim + 1,
                               hidden_size=config.hid_dim)
        self.Wy = nn.Parameter(torch.ones(config.emb_dim, config.hid_dim) * 0.0)
        self.Wh = torch.eye(config.hid_dim, requires_grad=True)
        self.Wt = torch.ones((1, config.hid_dim), requires_grad=True)  # * 1e-3
        self.Vy = nn.Parameter(torch.ones(config.hid_dim, self.n_class) * 1e-3)
        self.Vt = nn.Parameter(torch.ones(config.hid_dim, 1) * 1e-3)
        self.wt = nn.Parameter(torch.tensor(1.0))
        self.bh = nn.Parameter(torch.log(torch.ones(1, 1)))
        self.bk = nn.Parameter(torch.ones((1, self.n_class)) * 0.0)
        self.lstm = nn.LSTM(input_size=config.event_class + 2,
                            hidden_size=config.hid_dim,
                            batch_first=True,
                            bidirectional=False)
        self.gru = nn.GRU(input_size=12,
                          hidden_size=config.hid_dim,
                          batch_first=True)
        self.mlp = nn.Linear(in_features=config.hid_dim, out_features=config.mlp_dim)
        self.mlp_drop = nn.Dropout(p=config.dropout)

        self.event_linear = nn.Linear(in_features=config.mlp_dim, out_features=config.event_class)
        self.time_linear = nn.Linear(in_features=config.mlp_dim, out_features=1)
        self.set_criterion()

    def set_optimizer(self, total_step, use_bert=False):
        if use_bert:
            self.optimizer = BertAdam(params=self.parameters(),
                                      lr=self.config.lr,
                                      warmup=0.1,
                                      t_total=total_step)
        else:
            self.optimizer = Adam(self.parameters(), lr=self.config.lr)

    def set_criterion(self):
        self.event_criterion = nn.CrossEntropyLoss()
        if self.config.model == 'astro':
            self.intensity_w = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
            self.intensity_b = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
            self.time_criterion = self.astroLoss
        else:
            self.time_criterion = nn.MSELoss()

    def astroLoss(self, hidden_j, time_duration):
        loss = torch.mean(hidden_j + self.intensity_w * time_duration + self.intensity_b +
                          (torch.exp(hidden_j + self.intensity_b) -
                           torch.exp(
                               hidden_j + self.intensity_w * time_duration + self.intensity_b)) / self.intensity_w)
        return -loss

    def forward(self, input_time, event_input, gcn_out):
        # event ont-hot input
        event_onthot = torch.nn.functional.one_hot(event_input, num_classes=self.n_class)
        event_onthot = event_onthot.float()
        event_embedding = self.embed(event_onthot)

        # gcn input
        input_event = event_input.reshape(-1)
        out = torch.index_select(gcn_out, 0, input_event)
        out = out.reshape(input_time.shape[0], input_time.shape[1], -1)

        lstm_input = torch.cat((out, input_time.unsqueeze(-1)), dim=-1)
        lstm_input = torch.cat((lstm_input, event_embedding), dim=-1)
        hidden_state, _ = self.lstm(lstm_input)
        mlp_output = torch.tanh(self.mlp(hidden_state[:, -1, :]))
        mlp_output = self.mlp_drop(mlp_output)
        event_out = self.event_linear(mlp_output)
        time_out = self.time_linear(mlp_output)

        return event_out, time_out

    def dispatch(self, tensors):
        for i in range(len(tensors)):
            tensors[i] = tensors[i].contiguous()
        return tensors

    def train_batch(self, batch, out, device):
        time_tensor, event_tensor = batch
        time_tensor.to(device)
        event_tensor.to(device)

        # TODO 写入的duration 有误
        time_input, time_duration = self.dispatch([time_tensor[:, :-1], time_tensor[:, -1]])
        event_input, event_target = self.dispatch([event_tensor[:, :-1], event_tensor[:, -1]])
        event_out, time_out = self.forward(time_input, event_input, out)

        # TODO 输入应该为时间差
        loss1 = self.time_criterion(time_out.view(-1), time_duration.view(-1))
        loss2 = self.event_criterion(event_out.view(-1, self.n_class), event_target.view(-1))
        loss = self.config.alpha * loss1 + loss2

        loss.backward(retain_graph=True)

        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss1.item(), loss2.item(), loss.item()

    def predict(self, batch, out, device):
        time_tensor, event_tensor = batch
        time_tensor.to(device)
        event_tensor.to(device)
        # time_tensor.shape = (64,11)
        time_input, time_duration = self.dispatch([time_tensor[:, :-1], time_tensor[:, 1:]])
        # time_input.shape = (64,10),time_target.shape = (64)
        event_input, event_target = self.dispatch([event_tensor[:, :-1], event_tensor[:, -1]])

        event_out, time_out = self.forward(time_input, event_input, out)
        x = torch.log(torch.exp(time_out + self.intensity_b) + self.intensity_w * np.log(2)) - (
                time_out + self.intensity_b)
        x = torch.squeeze(x)
        duration = x / torch.squeeze(self.intensity_w)
        time_pred = duration  # + time_input[:,-1]

        event_pred = np.argmax(event_out.detach().numpy(), axis=-1)
        time_pred = np.array(time_pred.detach().numpy())

        return time_pred, event_pred
