import numpy as np
import matplotlib.pyplot as plt 
from typing import Iterable, List
import torch
from torchgeo.datasets import unbind_samples

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

 
def plot_results(hist,acc_fns):
    plt.figure(figsize=(15,5))
    
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
    
    for k, acc in enumerate(np.array(hist['val_metrics']).T):
        print(f'Validation\t{acc_fns[k].__name__}\tMaximum, Mean = {round(max(acc), 3)}\t{round(np.mean(acc),3)}')
        plt.subplot(int(np.ceil(acc_fns.__len__()/2)),2,k+1)
        plt.plot(np.array(acc), label=acc_fns[k].__name__)
        plt.legend()
        
    # vloss=hist['val_loss']
    # plt.subplot(322)
    # plt.plot(hist['val_loss'], label='Validation loss')
    # plt.legend()
    # plt.title(f'Minimum Validation loss= {round(min(vloss), 3)}, Mean Validation loss= {round(np.mean(vloss),3)}')


def plot_training(
    training_losses,
    validation_losses,
    learning_rate,
    gaussian=True,
    sigma=2,
    figsize=(8, 6),
):
    """
    Returns a loss plot with training loss, validation loss and learning rate.
    """

    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from scipy.ndimage import gaussian_filter

    list_len = len(training_losses)
    x_range = list(range(1, list_len + 1))  # number of x values

    fig = plt.figure(figsize=figsize)
    grid = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)

    subfig1 = fig.add_subplot(grid[0, 0])
    subfig2 = fig.add_subplot(grid[0, 1])

    subfigures = fig.get_axes()

    for i, subfig in enumerate(subfigures, start=1):
        subfig.spines["top"].set_visible(False)
        subfig.spines["right"].set_visible(False)

    if gaussian:
        training_losses_gauss = gaussian_filter(training_losses, sigma=sigma)
        validation_losses_gauss = gaussian_filter(validation_losses, sigma=sigma)

        linestyle_original = "."
        color_original_train = "lightcoral"
        color_original_valid = "lightgreen"
        color_smooth_train = "red"
        color_smooth_valid = "green"
        alpha = 0.25
    else:
        linestyle_original = "-"
        color_original_train = "red"
        color_original_valid = "green"
        alpha = 1.0

    # Subfig 1
    subfig1.plot(
        x_range,
        training_losses,
        linestyle_original,
        color=color_original_train,
        label="Training",
        alpha=alpha,
    )
    subfig1.plot(
        x_range,
        validation_losses,
        linestyle_original,
        color=color_original_valid,
        label="Validation",
        alpha=alpha,
    )
    if gaussian:
        subfig1.plot(
            x_range,
            training_losses_gauss,
            "-",
            color=color_smooth_train,
            label="Training",
            alpha=0.75,
        )
        subfig1.plot(
            x_range,
            validation_losses_gauss,
            "-",
            color=color_smooth_valid,
            label="Validation",
            alpha=0.75,
        )
    subfig1.title.set_text("Training & validation loss")
    subfig1.set_xlabel("Epoch")
    subfig1.set_ylabel("Loss")

    subfig1.legend(loc="upper right")

    # Subfig 2
    subfig2.plot(x_range, learning_rate, color="black")
    subfig2.title.set_text("Learning rate")
    subfig2.set_xlabel("Epoch")
    subfig2.set_ylabel("LR")

    # return fig