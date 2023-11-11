# we shall have predictions with shape (N, C, d1, d2) and we have masks with the shape (N, 1, d1, d2). For the loss function, 
# normally the Cross Entropy Loss should work, but it requires the mask to have shape (N, d1, d2). In this case, we will need to squeeze our second dimension manually.

# Additionally, we will create two accuracy functions. 
# The overall accuracy, used in the original paper and the intersect over union. 
# Usually when we have masks with unbalanced amount pixels in each class, the overall accuracy will result in unrealistic values. 
# In this case, the OA should be avoided, but it is left here for comparison with the original paper.

# The overall accuracy is calculated manually by adding all the matches and dividing by the number of elements in the batch. 
# The IoU is also known as Jaccard Index and it is available in Sklearn package. 
# The Pytorch’s cross entropy is used for loss, with a minor adjustment in the target’s shape. 
# After all the necessary adjustments the functions are defined as:


import torch
from sklearn.metrics import jaccard_score, f1_score, classification_report, precision_recall_fscore_support,precision_score,recall_score, accuracy_score

def loss(p, t):    
    return torch.nn.functional.cross_entropy(p, t.squeeze())

def iou(y, pred):
    return jaccard_score(y.reshape(-1), pred.reshape(-1), zero_division=1.)  

def f1s(y, pred):
    return f1_score(y.reshape(-1), pred.reshape(-1), zero_division=1.) 

def classrep(y, pred):
    return classification_report(y.reshape(-1), pred.reshape(-1), zero_division=1.,digits=3)

def uacc_pacc_f1s_sup(y, pred):
    return precision_recall_fscore_support(y.reshape(-1), pred.reshape(-1), zero_division=1.)

##User Accuracy
def uacc(y, pred):
    return precision_score(y.reshape(-1), pred.reshape(-1), zero_division=1.)

##Producer Accuracy
def pacc(y, pred):
    return recall_score(y.reshape(-1), pred.reshape(-1), zero_division=1.)

def oacc(y, pred):
    return accuracy_score(y.reshape(-1), pred.reshape(-1))


#Calculate Test area metrics for all batches in Test_Dataloader
def Tmetrics(model,Test_dataloader,acc_fns,devices):
    cuda_model = model.cuda(device=devices)
    acc = [0.] * len(acc_fns)
    # accum_valloss=0 
    with torch.no_grad():
        for batch in Test_dataloader:             
            X = batch['image'].type(torch.float32).cuda(device=devices)
            y = batch['mask'].type(torch.long).cuda(device=devices)
            pred = cuda_model(X)
        # accum_valloss += float(loss(pred, y)) / len(Test_dataloader)
            for i, acc_fn in enumerate(acc_fns):
                acc[i] = float(acc[i] + acc_fn(y.cpu(), pred.argmax(1).cpu())/len(Test_dataloader))    
    
    return acc