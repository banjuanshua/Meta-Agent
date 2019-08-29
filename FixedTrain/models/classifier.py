import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple
from models.function import relation_loss

def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.

    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size() + torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)

    return encoded_indicies

class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        n_p = 512
        # n_p = 4608
        self.n_p = n_p

        self.encoder = nn.Linear(n_p, 512)
        self.bn_coder = nn.BatchNorm1d(512)
        self.decoder = nn.Linear(512, n_p)

        self.fc_gen1 = nn.Linear(n_p, 1024)
        self.bn_gen = nn.BatchNorm1d(1024)
        self.fc_gen2 = nn.Linear(1024, 1)

        self.lr = nn.Parameter(torch.ones(n_p).float())
        self.scale = nn.Parameter(torch.FloatTensor([1.0]))


    def self_coder(self, x):
        x = self.encoder(x)
        x = F.leaky_relu(x, inplace=True)
        x = x.permute(0, 2, 1)
        x = self.bn_coder(x)
        x = x.permute(0, 2, 1)
        x = self.decoder(x)
        return x


    def reproject(self, support):
        pass


    def eval_genera(self, x):
        x = F.relu(self.fc_gen1(x))
        x = x.permute(0, 2, 1)
        x = self.bn_gen(x)
        x = x.permute(0, 2, 1)
        x = F.relu(self.fc_gen2(x))
        x = x.mean()
        return x


    def forward(self, prototypes, masktypes, query, support, support_labels, n_way, n_shot, normalize=True):
        # bs * n * m * 16000

        bs = prototypes.shape[0]
        n = query.shape[1]
        m = prototypes.shape[1]
        dims = prototypes.shape[2]

        query = query.unsqueeze(2).expand(bs, n, m, dims)
        prototypes = prototypes.unsqueeze(1).expand(bs, n, m, dims)

        # print('query', query.shape)
        # print('proto', prototypes.shape)

        logits = torch.pow(query-prototypes, 2).sum(dim=-1)

        # print('logits', logits.shape)


        if normalize:
            logits = logits / dims
        return self.scale * logits


    def log_prob(self, value, proto, sigma):
        var = (sigma ** 2)
        log_scale = sigma.log()
        dist = -((value - proto) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))
        return dist.mean(dim=1)


class Reproject(nn.Module):
    def __init__(self, n_way, feature_dim):
        super(Reproject, self).__init__()
        self.p_project = nn.Parameter(torch.ones(n_way, feature_dim).float())
        self.fc = nn.Linear(feature_dim, n_way)


    def forward(self, x):
        x = x * self.p_project
        logits = self.fc(x)
        return logits

    def reproject(self, x):
        return x * self.p_project.data



def head_test():
    head = Agent(10080)
    prototypes = torch.ones(2, 5, 10080) * 0.1
    prototypes[:,1,:] = 2 * 0.1
    prototypes[:, 2, :] = 3 * 0.1
    prototypes[:, 3, :] = 4 * 0.1
    prototypes[:, 4, :] = 5 * 0.1

    support = torch.ones(2, 75, 10080) * 0.1
    sigma = torch.ones(2, 5, 1) * 0.1
    labels = torch.zeros(8, 75, 1)
    labels = labels.long()

    param = namedtuple('param', ['eps', 'train_way'])
    opt = param(0.1, 5)

    logits = head(prototypes, support)

    print('logits', logits[0][0], logits.shape)
    print('softmax', F.softmax(logits)[0][0])


if __name__ == '__main__':
    head_test()
