import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple
from models.function import relation_loss
from models.classification_heads import *

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
    def __init__(self, bs, n_way, feature_dim):
        super(Agent, self).__init__()
        n_p = 16000
        self.n_p = n_p


        self.fc1 = nn.Linear(feature_dim, 128)
        self.bn = nn.BatchNorm1d(n_way)
        self.fc2 = nn.Linear(128, 1)

        self.scale = nn.Parameter(torch.ones(1).float())

        self.mask = nn.Parameter(torch.ones(bs, n_way, feature_dim).float())
        self.lr = nn.Parameter(torch.ones(16000).float())

    def get_proto(self, query, support, support_labels, n_way, n_shot):
        tasks_per_batch = query.size(0)
        n_support = support.size(1)

        assert (query.dim() == 3)
        assert (support.dim() == 3)
        assert (query.size(0) == support.size(0) and query.size(2) == support.size(2))
        assert (n_support == n_way * n_shot)  # n_support must equal to n_way * n_shot

        support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way)
        support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)

        # From:
        # https://github.com/gidariss/FewShotWithoutForgetting/blob/master/architectures/PrototypicalNetworksHead.py
        # ************************* Compute Prototypes **************************
        labels_train_transposed = support_labels_one_hot.transpose(1, 2)
        # Batch matrix multiplication:
        #   prototypes = labels_train_transposed * features_train ==>
        #   [batch_size x nKnovel x num_channels] =
        #       [batch_size x nKnovel x num_train_examples] * [batch_size * num_train_examples * num_channels]
        prototypes = torch.bmm(labels_train_transposed, support)
        # Divide with the number of examples per novel category.
        prototypes = prototypes.div(
            labels_train_transposed.sum(dim=2, keepdim=True).expand_as(prototypes)
        )

        return prototypes

    def forward(self, support, query, support_labels, n_way, n_shot, normalize=True):
        # bs * n * m * 16000

        classtypes = self.get_proto(query, support, support_labels, n_way, n_shot)
        # classtypes = classtypes * self.mask

        constraint = self.fc1(classtypes)
        constraint = self.bn(constraint)
        # print('bn cons', constraint[0][0])
        # constraint = self.fc2(constraint) / 128
        constraint = F.sigmoid(constraint).mean(dim=-1)
        # constraint = (constraint ** 2).mean(dim=-1)
        # constraint = constraint.squeeze(dim=-1)
        # print(constraint.shape)
        # constraint = (F.sigmoid(constraint)).mean(dim=-1)

        bs = classtypes.shape[0]
        n = query.shape[1]
        m = classtypes.shape[1]
        dims = classtypes.shape[2]

        query = query.unsqueeze(2).expand(bs, n, m, dims)
        classtypes = classtypes.unsqueeze(1).expand(bs, n, m, dims)
        # mask = self.mask.unsqueeze(1).expand(bs, n, m, dims)
        constraint = constraint.unsqueeze(1).expand(bs, n, n_way)

        # print('query', query.shape)
        # print('proto', classtypes.shape)
        # print('mask', mask.shape)

        # query = query * mask
        logits = torch.pow(query-classtypes, 2).sum(dim=-1)

        # print('logits', logits.shape)


        if normalize:
            logits = logits / dims

        # print('cons', constraint.shape)
        # print('logtis', logits.shape)
        # print('cons', constraint[0][0])
        # print('logits', logits[0][0])
        logits = torch.exp(logits-constraint)
        # sss

        # print('end', logits[0][0])


        return self.scale * logits


class ProtoHead(nn.Module):
    def __init__(self):
        super(ProtoHead, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([1.0]))

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
        # print('mask', mask.shape)

        if type(masktypes) != int:
            masktypes = masktypes.unsqueeze(1).expand(bs, n, m, dims)
            query = query * masktypes
            prototypes = prototypes * masktypes
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



def head_test():
    head = ProtoHead(10080)
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
