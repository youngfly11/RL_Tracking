import torch
import torch.nn as nn

import os
import shutil
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform(m.weight.data, gain=np.sqrt(2.0))

def accuracy(pred, label):
    """
    :param pred: the output from network
    :param label: groundtruth
    :return: accuracy
    """

    pred_np = pred.cpu().data.numpy()
    pred_np = np.squeeze(pred_np)

    label_np = label.cpu().data.numpy()
    label_np = np.squeeze(label_np)

    pred_np = pred_np >0.5
    acc = np.mean(pred_np==label_np)
    return acc


def save_checkpoint(state, is_best, epoch, filename='checkpoint.pth.tar',dir=None):

    if not os.path.exists(dir):
        os.makedirs(dir)


    # every ten epoch to save a checkpoint
    torch.save(state, os.path.join(dir, 'latest.pth.tar'))

    if (epoch) // 10 == 0 or is_best:
        torch.save(state, os.path.join(dir, filename))

        shutil.copyfile(os.path.join(dir, filename),
                        os.path.join(dir, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, lr, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch<=50:
        lr = 0.01
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    if epoch > 50 and epoch < 100:
    # if epoch > 50 and epoch < 100:
    # if epoch > 50 and epoch < 100:
        lr = 0.005
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif epoch > 100:
        lr = 0.001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

class AverageAccuracy(object):

    def __init__(self, threshold):
        super(AverageAccuracy, self).__init__()
        self.reset()
        self.elp = np.array([1e-8]*14,dtype=np.float64)
        self.threshold = threshold

    def reset(self):

        self.pred_dise = np.array([0]*14, dtype=np.float64)
        self.pred_heth = np.array([0]*14, dtype=np.float64)
        self.true_dise = np.array([0]*14, dtype=np.float64)
        self.true_heth = np.array([0]*14, dtype=np.float64)
        self.total_corr = np.array([0]*14, dtype=np.float64)
        self.num_sample = 0

    def update(self, output, target):

        num_sample, num_correct, num_dise_pred, num_dise_tar, num_heth_pred, num_heth_tar = \
            self.calculate_accuracy(output=output,target=target)
        self.num_sample += num_sample
        self.pred_dise += num_dise_pred
        self.true_dise += num_dise_tar
        self.pred_heth += num_heth_pred
        self.true_heth += num_heth_tar
        self.total_corr += num_correct
        self.avg_total_corr = self.total_corr / float(self.num_sample)
        self.avg_pred_dise = self.pred_dise / (self.true_dise + self.elp)
        self.avg_pred_heth = self.pred_heth / (self.true_heth + self.elp)
        self.acc = np.mean(self.avg_total_corr)

    def calculate_accuracy(self, output, target):
        """
        to calculate three different acc, disease_acc, and no_disease_acc and total_acc

        output: the numpy format, N*14
        target: numpy format  N*14
        """
        n, _ = output.shape
        output = 1 * (output > self.threshold) ## change the true false label in to 0,1
        num_correct = np.sum(output==target,axis=0)
        num_dise_pred = np.sum(output*target,0)
        num_dise_tar = np.sum(target,0)
        num_heth_pred = np.sum((1-output)*(1-target),axis=0)
        num_heth_tar = np.sum((1-target), axis=0)
        num_sample = n

        return (num_sample, num_correct, num_dise_pred, num_dise_tar, num_heth_pred, num_heth_tar)


def infofmt(data, datatype):

    infofmt = '{datatype}:'\
              "class0:{class0:.3f}\t" \
              "class1:{class1:.3f}\t" \
              "class2:{class2:.3f}\t" \
              "class3:{class3:.3f}\t" \
              "class4:{class4:.3f}\t" \
              "class5:{class5:.3f}\t" \
              "class6:{class6:.3f}\t" \
              "class7:{class7:.3f}\t" \
              "class8:{class8:.3f}\t" \
              "class9:{class9:.3f}\t" \
              "class10:{class10:.3f}\t" \
              "class11:{class11:.3f}\t" \
              "class12:{class12:.3f}\t" \
              "class13:{class13:.3f}\t"

    infodict = dict(
        datatype=datatype,
        class0=data[0],
        class1=data[1],
        class2=data[2],
        class3=data[3],
        class4=data[4],
        class5=data[5],
        class6=data[6],
        class7=data[7],
        class8=data[8],
        class9=data[9],
        class10=data[10],
        class11=data[11],
        class12=data[12],
        class13=data[13]
    )
    info1 = infofmt.format(**infodict)
    return info1


def extract_features(x, cnn):
    """
    Use the CNN to extract features from the input image x.

    Inputs:
    - x: A PyTorch Variable of shape (N, C, H, W) holding a minibatch of images that
      will be fed to the CNN.
    - cnn: A PyTorch model that we will use to extract features.

    Returns:
    - features: A list of feature for the input images x extracted using the cnn model.
      features[i] is a PyTorch Variable of shape (N, C_i, H_i, W_i); recall that features
      from different layers of the network may have different numbers of channels (C_i) and
      spatial dimensions (H_i, W_i).
    """
    features_mean = {}
    prev_feat = x
    for i, module in enumerate(cnn._modules.values()):
        next_feat = module(prev_feat)
        print(i, next_feat.size())
        features_mean[i] = next_feat.mean()
        prev_feat = next_feat
    return features_mean

if __name__=='__main__':

    x = np.random.randn(4,4)

    y = np.array([[1,0,1,0],[0,1,1,0],[1,1,1,1],[0,0,1,1]])
    acc = AverageAccuracy(threshold=0.5)
    acc.update(output=x, target=y)
