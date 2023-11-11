from typing import List, Optional, Callable
from torch.utils.data import DataLoader
import torch

# The training function should receive 
# the number of epochs, 
# the model, 
# the dataloaders, 
# the loss function (to be optimized) 
# the accuracy function (to assess the results), 
# the optimizer (that will adjust the parameters of the model in the correct direction) and 
# the transformations to be applied to each batch.

def train_loop(
    epochs: int, 
    train_dl: DataLoader, 
    val_dl: Optional[DataLoader], 
    model: torch.nn.Module, 
    loss_fn: Callable, 
    optimizer: torch.optim.Optimizer, 
    acc_fns: Optional[List]=None, 
    batch_tfms: Optional[Callable]=None,
    devices: Optional[Callable]="cuda:0"
):
    
    cuda_model = model.cuda(device=devices)
    res = {'train_loss': [],'val_loss': [], 'val_metrics': []}
    for epoch in range(epochs):
        accum_trainloss = 0
        for batch in train_dl:

            if batch_tfms is not None:
                batch = batch_tfms(batch)

            X = batch['image'].cuda(device=devices)
            y = batch['mask'].type(torch.long).cuda(device=devices)
            pred = cuda_model(X)
            loss = loss_fn(pred, y)

            # BackProp
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update the accum loss
            accum_trainloss += float(loss) / len(train_dl)

        # Testing against the validation dataset
        if acc_fns is not None and val_dl is not None:
            # reset the accuracies metrics
            acc = [0.] * len(acc_fns)
            accum_valloss=0         
            with torch.no_grad():
                for batch in val_dl:

                    if batch_tfms is not None:
                        batch = batch_tfms(batch)                    

                    X = batch['image'].type(torch.float32).cuda(device=devices)
                    y = batch['mask'].type(torch.long).cuda(device=devices)

                    pred = cuda_model(X)
                    accum_valloss += float(loss_fn(pred, y)) / len(val_dl)
                    
                    for i, acc_fn in enumerate(acc_fns):
                        acc[i] = float(acc[i] + acc_fn(y.cpu(), pred.argmax(1).cpu())/len(val_dl))

            # at the end of the epoch, print the errors, etc.
            print(f'Epoch {epoch}: Train Loss={accum_trainloss:.5f}, validation Loss={accum_valloss:.5f} - Accs={[round(a, 3) for a in acc]}')
        else:

            print(f'Epoch {epoch}: Train Loss={accum_trainloss:.5f}')
        
        res['train_loss'].append(accum_trainloss)
        res['val_loss'].append(accum_valloss)
        res['val_metrics'].append(acc)

    return res