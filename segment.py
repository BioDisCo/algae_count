import cv2
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cellpose_omni import io, transforms
from omnipose.utils import normalize99
from cellpose_omni import models, core
from cellpose_omni import models
from cellpose_omni.models import MODEL_NAMES
import time
from cellpose_omni import plot
import omnipose
import pandas as pd

CSV_FILENAME = 'cell_nr.csv'
RESULTS_DIR = 'results'
TIFF_DIR = "tiffs/"  # place all tiffs there

mpl.rcParams['figure.dpi'] = 600
plt.style.use('dark_background')
use_GPU = core.use_gpu()
print('GPU ' + 'activated' if use_GPU else 'deactivated')

def read_tiff_images(directory):
    images = {}
    for filename in os.listdir(directory):
        basename = filename.split('.')[0]
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is not None:
                images[basename] = image
            else:
                print(f"Failed to read image: {image_path}")
    return images

imgs = read_tiff_images(TIFF_DIR)
nimg = len(imgs)

print(f'{nimg} images found')
for i in imgs.values():
    print('image shape:',i.shape)
    print('data type:',i.dtype)
    print('data range: min {}, max {}\n'.format(i.min(),i.max()))

for k in imgs.keys():
    img = transforms.move_min_dim(imgs[k]) # move the channel dimension last
    if len(img.shape) > 2:
        imgs[k] = np.mean(img,axis=-1) # grayscale 
        
    imgs[k] = normalize99(imgs[k])
    print('new shape: ', imgs[k].shape)


print(f'available models: {MODEL_NAMES}')
model_name = 'bact_phase_omni'

# use this to get models from the web
# model = models.CellposeModel(gpu=use_GPU, model_type=model_name)

# and this to get them locally
model = models.CellposeModel(gpu=use_GPU, pretrained_model='models/bact_phase_omnitorch_0')

chans = [0,0]  #this means segment based on first channel, no second channel 

# define parameters
params = {'channels':chans, # always define this with the model
          'rescale': None, # upscale or downscale your images, None = no rescaling 
          'mask_threshold': -1, # erode or dilate masks with higher or lower values 
          'flow_threshold': 0, # default is .4, but only needed if there are spurious masks to clean up; slows down output
          'transparency': True, # transparency in flow output
          'omni': True, # we can turn off Omnipose mask reconstruction, not advised 
          'cluster': True, # use DBSCAN clustering
          'resample': True, # whether or not to run dynamics on rescaled grid or original grid 
          # 'verbose': False, # turn on if you want to see more output 
          'tile': False, # average the outputs from flipped (augmented) images; slower, usually not needed 
          'niter': None, # None lets Omnipose calculate # of Euler iterations (usually <20) but you can tune it for over/under segmentation 
          'augment': False, # Can optionally rotate the image and average outputs, usually not needed 
          'affinity_seg': False, # new feature, stay tuned...
         }

tic = time.time()
masks, flows, styles = model.eval([img for img in imgs.values()], **params)
net_time = time.time() - tic
print('total segmentation time: {}s'.format(net_time))

# write results
os.makedirs(RESULTS_DIR, exist_ok=True)

cells_nr = {}
for idx, img_name in enumerate(imgs.keys()):
    img = imgs[img_name]
    maski = masks[idx] # get masks
    bdi = flows[idx][-1] # get boundaries
    flowi = flows[idx][0] # get RGB flows

    # show it for debugging
    fig = plt.figure(figsize=(4,1), dpi=4000)
    fig.patch.set_facecolor([0]*4)
    plot.show_segmentation(
        fig,
        omnipose.utils.normalize99(img), 
        maski,
        flowi,
        bdi,
        channels=chans,
        omni=True,
        interpolation=None)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    file_path = os.path.join(RESULTS_DIR, f"debug-{img_name}.png")
    plt.savefig(file_path)
    plt.close()

    fig = plt.figure(figsize=(1,1), dpi=4000)
    fig.patch.set_facecolor([0]*4)
    plt.axis('off')
    plt.imshow(flowi)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    file_path = os.path.join(RESULTS_DIR, f"seg-{img_name}.png")
    plt.savefig(file_path)
    plt.close()

    labels = set(maski.ravel())
    cells = len(labels) - 1  # remove the background label
    cells_nr[img_name] = cells


# export to CSV
df = pd.DataFrame({'img': cells_nr.keys(), 'cells_nr': cells_nr.values()})
file_path = os.path.join(RESULTS_DIR, CSV_FILENAME)
df.to_csv(file_path, index=False)
