from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import numpy as np



def metric_auroc(label_prob, label):
    '''
    Arg:
        label_prob: all label prob in shape 11210*14, format: numpy
        label: all label ground truth label in shape 11210*14, format: numpy

    return:
        roc_score: N*14, which is the area under ROC
    '''

    # num_sample, num_diseases = label_prob[:,None].shape
    label_prob =  label_prob[:,None]
    num_sample, num_diseases = label_prob.shape
    label = label[:, None]
    roc_score = []
    for diseases_idx in range(num_diseases):
        y_true = label[:, diseases_idx]
        y_score = label_prob[:, diseases_idx]
        try:
            diseases_roc = roc_auc_score(y_true=y_true, y_score=y_score)
        except ValueError:
            diseases_roc = 0.0
        roc_score.append(diseases_roc)

    return roc_score


def fbeta_score(label_prob, label, beta=2):
    '''
    Arg:
        label_prob: all label prob in shape 11210*14, format: numpy
        label: all label ground truth label in shape 11210*14, format: numpy

    return:
        fbeta_score: N*14, which is the metric in PR curve
    '''
    # num_sample, num_diseases = label_prob.shape
    label_prob = label_prob[:,None]
    label= label[:,None]
    num_sample, num_diseases = label_prob.shape
    f1_score = []
    f2_score = []
    for diseases_idx in range(num_diseases):
        y_true = label[:, diseases_idx]
        y_score = label_prob[:, diseases_idx]
        precision, recall, threshold = precision_recall_curve(y_true, y_score)
        prec_mean = np.mean(precision)
        recall_mean = np.mean(recall)

        ## calculate the f1 score
        f1 = score_func(prec_mean, recall_mean, beta=1)
        f1_score.append(f1)

        ## calculate the f2 score
        f2 = score_func(prec_mean, recall_mean, beta=2)
        f2_score.append(f2)
    return f1_score, f2_score

def score_func(prec,recall,beta=1):

    fbeta = ((1+beta**2) * prec * recall)/float(beta**2*prec+recall)
    return fbeta

if __name__=='__main__':

    np.random.seed(10)
    prob = np.random.rand(4,4)
    print prob
    gt = np.array([[0,1,1,0],[1,0,1,0],[0,1,1,1],[0,0,0,1]])

    f1_score = metric_auroc(label_prob=prob, label=gt)

