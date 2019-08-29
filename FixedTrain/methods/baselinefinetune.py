import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from models.classifier import Agent
from models.classification_heads import *
from models.utils import *


class BaselineFinetune(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, loss_type = "softmax"):
        super(BaselineFinetune, self).__init__( model_func,  n_way, n_support)
        self.loss_type = loss_type
        self.agent = Agent().cuda()


    def set_forward(self,x,is_feature = True):
        return self.set_forward_adaptation(x,is_feature); #Baseline always do adaptation


    def atrain(self, support, query, labels_support, labels_query, is_feature = True):
        eps = 0.1
        assert is_feature == True, 'Baseline only support testing with feature'

        agent = self.agent
        optimizer = self.optimizer

        re_epochs = 10

        # support = agent.self_coder(support)
        # query = agent.self_coder(query)


        classtypes = agent.get_proto(query, support, labels_support, self.n_way, self.n_support)

        for _ in range(10):
            classtypes = Variable(classtypes.data, requires_grad=True)
            # rl_loss = relation_loss(classtypes, emb_support)
            ev = agent.eval_genera(classtypes)
            loss = ev + 0.0001  # + rl_loss
            print(loss.data)
            classtypes.grad = None
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            # grad = agent.self_coder(classtypes.grad)
            grad = classtypes.grad
            # classtypes = classtypes - grad.data
            classtypes = classtypes - agent.lr.data * grad.data
            # print('grad', grad.data[0][0])


        logit_query = agent(classtypes, 0, query, support, labels_support,
                               self.n_way, self.n_support)


        smoothed_one_hot = one_hot(labels_query.reshape(-1), self.n_way)
        smoothed_one_hot = smoothed_one_hot * (1 - eps) + (1 - smoothed_one_hot) * eps / (self.n_way - 1)
        log_prb = F.log_softmax(logit_query.reshape(-1, self.n_way), dim=1)
        cls_loss = -(smoothed_one_hot * log_prb).sum(dim=1)
        cls_loss = cls_loss.mean()

        optimizer.zero_grad()
        cls_loss.backward()
        optimizer.step()

        acc = count_accuracy(logit_query.reshape(-1, self.n_way), labels_query.reshape(-1))


        print('loss', cls_loss.item(), 'acc', acc.item())

        return logit_query



    def set_forward_loss(self,x):
        raise ValueError('Baseline predict on pretrained feature and do not support finetune backbone')
        

