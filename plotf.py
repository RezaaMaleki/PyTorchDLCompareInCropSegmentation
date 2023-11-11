import numpy as np
import matplotlib.pyplot as plt 
from typing import Iterable, List
import torch
from torchgeo.datasets import RasterDataset, stack_samples, unbind_samples, Sentinel2 , CDL

def plot_imgs(images: Iterable, axs: Iterable, chnls: List[int] = [2, 1, 0], bright: float = 3.):
    for img, ax in zip(images, axs):
        arr = torch.clamp(bright * img, min=0, max=1).numpy()
        rgb = arr.transpose(1, 2, 0)[:, :, chnls]
        ax.imshow(rgb)
        ax.axis('off')


def plot_msks(masks: Iterable, axs: Iterable):
    for mask, ax in zip(masks, axs):
        ax.imshow(mask.squeeze().numpy(), cmap='Oranges')
        ax.axis('off')

def plot_batch(batch: dict, bright: float = 3., cols: int = 4, width: int = 5, chnls: List[int] = [3, 2, 1]):

    # Get the samples and the number of items in the batch
    samples = unbind_samples(batch.copy())
    
    # if batch contains images and masks, the number of images will be doubled
    n = 2 * len(samples) if ('image' in batch) and ('mask' in batch) else len(samples)

    # calculate the number of rows in the grid
    rows = n//cols + (1 if n%cols != 0 else 0)

    # create a grid
    _, axs = plt.subplots(rows, cols, figsize=(cols*width, rows*width))  

    if ('image' in batch) and ('mask' in batch):
        # plot the images on the even axis
        plot_imgs(images=map(lambda x: x['image'], samples), axs=axs.reshape(-1)[::2], chnls=chnls, bright=bright) 

        # plot the masks on the odd axis
        plot_msks(masks=map(lambda x: x['mask'], samples), axs=axs.reshape(-1)[1::2]) 

    else:

        if 'image' in batch:
            plot_imgs(images=map(lambda x: x['image'], samples), axs=axs.reshape(-1), chnls=chnls, bright=bright) 
    
        elif 'mask' in batch:
            plot_msks(masks=map(lambda x: x['mask'], samples), axs=axs.reshape(-1)) 


def plot_compare(batch: dict, bright: float = 3., cols: int = 3, width: int = 5, chnls: List[int] = [3, 2, 1]):

    # Get the samples and the number of items in the batch
    samples = unbind_samples(batch.copy())
    
    # calculate the number of rows in the grid
    rows = len(samples)

    # create a grid
    _, axs = plt.subplots(rows, cols, figsize=(cols*width, rows*width))  

    if ('image' in batch) and ('mask' in batch) and ('pred' in batch):
        # plot the images on column number 1
        plot_imgs(images=map(lambda x: x['image'], samples), axs=axs.reshape(-1)[0::3] ,chnls=chnls, bright=bright) 
       
        # plot the masks on column number 2
        plot_msks(masks=map(lambda x: x['mask'], samples), axs=axs.reshape(-1)[1::3]) 
        
        # plot the prediction on column number 3
        plot_msks(masks=map(lambda x: x['pred'], samples), axs=axs.reshape(-1)[2::3])

    else:
        print ("there is not enough data")


 


def plot_results(hist):
    plt.figure(figsize=(15,5))
    
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
    
    acc=np.array(hist['val_metrics']).T[0]
    plt.subplot(321)
    plt.plot(np.array(hist['val_metrics']).T[0], label='Validation acc')
    plt.legend()
    plt.title(f'Maximum Validation acc= {round(max(acc), 3)}, Mean Validation acc= {round(np.mean(acc),3)}')

    IoU=np.array(hist['val_metrics']).T[1]
    plt.subplot(323)
    plt.plot(IoU, label='Validation IoU')
    plt.legend()
    plt.title(f'Maximum Validation IoU= {round(max(IoU), 3)}, Mean Validation IoU= {round(np.mean(IoU),3)}')

    f1s=np.array(hist['val_metrics']).T[2]
    plt.subplot(324)
    plt.plot(f1s, label='Validation F1-Score')
    plt.legend()
    plt.title(f'Maximum Validation F1-Score= {round(max(f1s), 3)}, Mean Validation F1-Score= {round(np.mean(f1s),3)}')

    vloss=hist['val_loss']
    plt.subplot(322)
    plt.plot(hist['val_loss'], label='Validation loss')
    plt.legend()
    plt.title(f'Minimum Validation loss= {round(min(vloss), 3)}, Mean Validation loss= {round(np.mean(vloss),3)}')
