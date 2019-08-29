import torch.optim
import torch.utils.data.sampler
from torch.autograd import Variable
import random

import configs
import data.feature_loader as feat_loader
from utils import *
from models.classifier import *


def agent_test(test_data_file, n_way=5, n_support=5, n_query=15):
    test_epochs = 600
    generation_epochs = 10

    class_list = test_data_file.keys()
    acc_all = []

    agent = Agent()
    # load state

    optimizer = torch.optim.SGD(agent.parameters(), lr=0.01, momentum=0.9, dampening=0.9,
                                weight_decay=0.001)

    for ep in range(test_epochs):

        select_class = random.sample(class_list,n_way)
        z_all  = []
        for cl in select_class:
            img_feat = test_data_file[cl]
            perm_ids = np.random.permutation(len(img_feat)).tolist()
            z_all.append( [ np.squeeze( img_feat[perm_ids[i]]) for i in range(n_support+n_query) ] )     # stack each batch

        z_all = torch.from_numpy(np.array(z_all) )

        support = z_all[:, :n_support]
        query = z_all[:, n_support:].contiguous().view(n_way * n_query, -1)
        y_support = np.repeat(range(n_way), n_support)
        y_query = np.repeat(range(n_way), n_query)

        labels_support =  torch.from_numpy(y_support)
        labels_query = torch.from_numpy(y_query)



        classtypes = support.mean(dim=2)
        classtypes = agent.self_coder(classtypes)
        query = agent.self_coder(query)

        for _ in range(generation_epochs):
            classtypes = Variable(classtypes.data, requires_grad=True)
            # rl_loss = relation_loss(classtypes, emb_support)
            ev = agent.eval_genera(classtypes)
            loss = ev + 0.0001  # + rl_loss
            # print(loss.data)
            classtypes.grad = None
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            # grad = agent.self_coder(classtypes.grad)
            grad = classtypes.grad
            # classtypes = classtypes - grad.data
            classtypes = classtypes - agent.lr.data * grad.data
            # print('grad', grad.data[0][0])

        logit_query = agent(classtypes, 0, query, support, labels_support,
                            n_way, n_support)


        acc = count_accuracy(logit_query.reshape(-1, n_way), labels_query.reshape(-1))
        acc_all.append(acc.item())

    acc_all = np.array(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    print('%d Test Acc = %4.2f%% +- %4.2f%%' % (ep, acc_mean, 1.96 * acc_std / np.sqrt(test_epochs)))
    acc_all = []





if __name__ == '__main__':
    params = parse_args('test')
    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot)


    checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        checkpoint_dir += '_aug'
    if not params.method in ['baseline', 'baseline++'] :
        checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)

    split = params.split
    if params.save_iter != -1:
        split_str = split + "_" +str(params.save_iter)
    else:
        split_str = split



    novel_file = os.path.join( checkpoint_dir.replace("checkpoints","features"), split_str +".hdf5")
    outs = ['./features/miniImagenet/ResNet34_baseline_aug/novel.hdf5',
            './features/miniImagenet/ResNet34_baseline_aug/base.hdf5',
            './features/miniImagenet/ResNet34_baseline_aug/val.hdf5']


    test_file = outs[0]
    test_data_file = feat_loader.init_loader(test_file)


    agent_test(test_data_file, n_query = 15, **few_shot_params)
