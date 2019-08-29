import numpy as np
import glob
import argparse
import backbone
import os
import time
import torch

def one_hot(y, num_class):
    return torch.zeros((len(y), num_class)).scatter_(1, y.unsqueeze(1), 1)

def DBindex(cl_data_file):
    class_list = cl_data_file.keys()
    cl_num= len(class_list)
    cl_means = []
    stds = []
    DBs = []
    for cl in class_list:
        cl_means.append( np.mean(cl_data_file[cl], axis = 0) )
        stds.append( np.sqrt(np.mean( np.sum(np.square( cl_data_file[cl] - cl_means[-1]), axis = 1))))

    mu_i = np.tile( np.expand_dims( np.array(cl_means), axis = 0), (len(class_list),1,1) )
    mu_j = np.transpose(mu_i,(1,0,2))
    mdists = np.sqrt(np.sum(np.square(mu_i - mu_j), axis = 2))

    for i in range(cl_num):
        DBs.append( np.max([ (stds[i]+ stds[j])/mdists[i,j]  for j in range(cl_num) if j != i ]) )
    return np.mean(DBs)

def sparsity(cl_data_file):
    class_list = cl_data_file.keys()
    cl_sparsity = []
    for cl in class_list:
        cl_sparsity.append(np.mean([np.sum(x!=0) for x in cl_data_file[cl] ])  )

    return np.mean(cl_sparsity)


model_dict = dict(
    Conv4=backbone.Conv4,
    Conv4S=backbone.Conv4S,
    Conv6=backbone.Conv6,
    ResNet10=backbone.ResNet10,
    ResNet18=backbone.ResNet18,
    ResNet34=backbone.ResNet34,
    ResNet50=backbone.ResNet50,
    ResNet101=backbone.ResNet101,
    resnet=backbone.resnet)


def parse_args(script):
    parser = argparse.ArgumentParser(description='few-shot script %s' % (script))
    parser.add_argument('--dataset', default='CUB', help='CUB/miniImagenet/cross/omniglot/cross_char')
    parser.add_argument('--model', default='Conv4',
                        help='model: Conv{4|6} / ResNet{10|18|34|50|101}')  # 50 and 101 are not used in the paper
    parser.add_argument('--method', default='baseline',
                        help='baseline/baseline++/protonet/matchingnet/relationnet{_softmax}/maml{_approx}')  # relationnet_softmax replace L2 norm with softmax to expedite training, maml_approx use first-order approximation in the gradient for efficiency
    parser.add_argument('--train_n_way', default=5, type=int,
                        help='class num to classify for training')  # baseline and baseline++ would ignore this parameter
    parser.add_argument('--test_n_way', default=5, type=int,
                        help='class num to classify for testing (validation) ')  # baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--n_shot', default=5, type=int,
                        help='number of labeled data in each class, same as n_support')  # baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--train_aug', action='store_true',
                        help='perform data augmentation or not during training ')  # still required for save_features.py and agtest.py to find the model path correctly

    if script == 'train':
        parser.add_argument('--num_classes', default=200, type=int,
                            help='total number of classes in softmax, only used in baseline')  # make it larger than the maximum label value in base class
        parser.add_argument('--save_freq', default=50, type=int, help='Save frequency')
        parser.add_argument('--start_epoch', default=0, type=int, help='Starting epoch')
        parser.add_argument('--stop_epoch', default=-1, type=int,
                            help='Stopping epoch')  # for meta-learning methods, each epoch contains 100 episodes. The default epoch number is dataset dependent. See train.py
        parser.add_argument('--resume', action='store_true',
                            help='continue from previous trained model with largest epoch')
        parser.add_argument('--warmup', action='store_true',
                            help='continue from baseline, neglected if resume is true')  # never used in the paper
    elif script == 'save_features':
        parser.add_argument('--split', default='novel',
                            help='base/val/novel')  # default novel, but you can also test base/val class accuracy if you want
        parser.add_argument('--save_iter', default=-1, type=int,
                            help='save feature from the model trained in x epoch, use the best model if x is -1')
    elif script == 'test':
        parser.add_argument('--split', default='novel',
                            help='base/val/novel')  # default novel, but you can also test base/val class accuracy if you want
        parser.add_argument('--save_iter', default=-1, type=int,
                            help='saved feature from the model trained in x epoch, use the best model if x is -1')
        parser.add_argument('--adaptation', action='store_true', help='further adaptation in test time or not')
    else:
        raise ValueError('Unknown script')

    return parser.parse_args()


def get_assigned_file(checkpoint_dir, num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file


def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist = [x for x in filelist if os.path.basename(x) != 'best_model.tar']
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file


def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)


def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)

def check_dir(path):
    '''
    Create directory if it does not exist.
        path:           Path of directory.
    '''
    if not os.path.exists(path):
        os.mkdir(path)

def count_accuracy(logits, label):
    pred = torch.argmax(logits, dim=1)#.view(-1)
    label = label.view(-1)
    accuracy = 100 * pred.eq(label).float().mean()
    return accuracy

class Timer():
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / float(p)
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

def log(log_file_path, string):
    '''
    Write one line of log into screen and file.
        log_file_path: Path of log file.
        string:        String to write in log file.
    '''
    with open(log_file_path, 'a+') as f:
        f.write(string + '\n')
        f.flush()
    print(string)

