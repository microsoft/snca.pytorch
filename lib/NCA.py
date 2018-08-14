import torch
from torch import nn
from torch.autograd import Function
import math

eps = 1e-8

class NCACrossEntropy(nn.Module): 
    ''' \sum_{j=C} log(p_{ij})
        Store all the labels of the dataset.
        Only pass the indexes of the training instances during forward. 
    '''
    def __init__(self, labels, margin=0):
        super(NCACrossEntropy, self).__init__()
        self.register_buffer('labels', torch.LongTensor(labels.size(0)))
        self.labels = labels
        self.margin = margin

    def forward(self, x, indexes):
        batchSize = x.size(0)
        n = x.size(1)
        exp = torch.exp(x)
        
        # labels for currect batch
        y = torch.index_select(self.labels, 0, indexes.data).view(batchSize, 1) 
        same = y.repeat(1, n).eq_(self.labels)

       # self prob exclusion, hack with memory for effeciency
        exp.data.scatter_(1, indexes.data.view(-1,1), 0)

        p = torch.mul(exp, same.float()).sum(dim=1)
        Z = exp.sum(dim=1)

        Z_exclude = Z - p
        p = p.div(math.exp(self.margin))
        Z = Z_exclude + p

        prob = torch.div(p, Z)
        prob_masked = torch.masked_select(prob, prob.ne(0))

        loss = prob_masked.log().sum(0)

        return - loss / batchSize

