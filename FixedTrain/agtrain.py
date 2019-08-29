from torch.autograd import Variable
import torch.optim
import torch.utils.data.sampler
import random

import configs
import data.feature_loader as feat_loader
from utils import *
from models.classifier import *



def sample_batch(cl_data_file, n_way, n_support, n_query, batch_size):
    class_list = cl_data_file.keys()

    support = []
    query = []
    labels_support = []
    labels_query = []

    for _ in range(batch_size):
        select_class = random.sample(class_list, n_way)
        z_all = []
        for cl in select_class:
            img_feat = cl_data_file[cl]
            perm_ids = np.random.permutation(len(img_feat)).tolist()
            z_all.append([np.squeeze(img_feat[perm_ids[i]]) for i in range(n_support + n_query)])  # stack each batch

        z_all = torch.from_numpy(np.array(z_all))
        emb_support = z_all[:, :n_support]
        emb_query = z_all[:, n_support:].contiguous().view(n_way * n_query, -1)
        emb_query = emb_query.numpy()

        y_support = np.repeat(range(n_way), n_support)
        y_query = np.repeat(range(n_way), n_query)

        emb_query = emb_query.tolist()
        y_query = y_query.tolist()

        arr = list(zip(emb_query, y_query))
        random.shuffle(arr)
        emb_query, y_query = zip(*arr)
        emb_query = np.array(emb_query)
        y_query = np.array(y_query)

        support.append(emb_support.unsqueeze(0))
        query.append(torch.from_numpy(emb_query).unsqueeze(0).float())
        labels_support.append(torch.from_numpy(y_support).unsqueeze(0))
        labels_query.append(torch.from_numpy(y_query).unsqueeze(0))



    support_4 = torch.cat(support, dim=0).cuda()
    query_3 = torch.cat(query, dim=0).cuda()
    labels_support = torch.cat(labels_support, dim=0).cuda()
    labels_query = torch.cat(labels_query, dim=0).cuda()

    return support_4, query_3, labels_support, labels_query


def agent_train(base_data_file, val_data_file, n_way = 5, n_support = 5, n_query = 15):
    acc_all = []
    epochs = int(1e6)
    val_epochs = 2000

    batch_size = 32
    eps = 0.1
    generation_epochs = 10



    agent = Agent().cuda()


    optimizer = torch.optim.SGD(agent.parameters(), lr=0.01, momentum=0.9, dampening=0.9,
                                     weight_decay=0.001)

    for i in range(epochs):
        agent.train()
        support, query, labels_support, labels_query = sample_batch(
            base_data_file, n_way, n_support, n_query, batch_size)

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

        smoothed_one_hot = one_hot(labels_query.reshape(-1), n_way)
        smoothed_one_hot = smoothed_one_hot * (1 - eps) + (1 - smoothed_one_hot) * eps / (n_way - 1)
        log_prb = F.log_softmax(logit_query.reshape(-1, n_way), dim=1)
        cls_loss = -(smoothed_one_hot * log_prb).sum(dim=1)
        cls_loss = cls_loss.mean()

        optimizer.zero_grad()
        cls_loss.backward()
        optimizer.step()

        acc = count_accuracy(logit_query.reshape(-1, n_way), labels_query.reshape(-1))
        acc_all.append(acc.item())

        # print('loss', round(cls_loss.item(),3), 'acc', acc.item())

        if i % 200 == 0 and i != 0:
            acc_all = np.array(acc_all)
            acc_mean = np.mean(acc_all)
            acc_std = np.std(acc_all)
            print('%d Train Acc = %4.2f%% +- %4.2f%%' % (i, acc_mean, 1.96 * acc_std / np.sqrt(epochs)))
            acc_all = []

        # Validation
        if i%4000==0 and i!=0:
            agent.eval()

            for i in range(val_epochs):
                support, query, labels_support, labels_query = sample_batch(
                    val_data_file, n_way, n_support, n_query, batch_size)

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

                # smoothed_one_hot = one_hot(labels_query.reshape(-1), n_way)
                # smoothed_one_hot = smoothed_one_hot * (1 - eps) + (1 - smoothed_one_hot) * eps / (n_way - 1)
                # log_prb = F.log_softmax(logit_query.reshape(-1, n_way), dim=1)
                # cls_loss = -(smoothed_one_hot * log_prb).sum(dim=1)
                # cls_loss = cls_loss.mean()

                acc = count_accuracy(logit_query.reshape(-1, n_way), labels_query.reshape(-1))
                acc_all.append(acc.item())

                # print('val loss', round(cls_loss.item(),3), 'val acc', acc.item())

            acc_all_val = np.array(acc_all)
            acc_mean_val= np.mean(acc_all_val)
            acc_std_val = np.std(acc_all_val)
            print('---- %d Val Acc = %4.2f%% +- %4.2f%%' % (i, acc_mean_val, 1.96 * acc_std_val / np.sqrt(epochs)))




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

    base_file = outs[1]
    val_file = outs[2]

    base_data_file = feat_loader.init_loader(base_file)
    val_data_file = feat_loader.init_loader(val_file)


    agent_train(base_data_file, val_data_file, n_query = 15, **few_shot_params)