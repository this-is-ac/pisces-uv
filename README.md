# pisces-uv


ffn 
-> demo.py will give outputs for 2 images
-> run_uw.py - get flow at Dh (512) and Dl (256) for all images in a folder, saved as .npy files 
-> visualize_flow.py - visualize a random flow file


llve
-> warp.py -> warping functions
-> utils -> utils for PhISHNet
-> train_PhiSHNet and train_PhishVideo.py -> training scripts
-> test_PhISHVideo.py -> test, use legacy=True for phishnet
-> slice.py - bilateral
-> perceptual_losses.py -> L_FID

-> model.py -> orig llve model
-> model_depthv2.py -> phishnet model
-> metrics.py
-> load_raw.py
-> evaluate.py -> ebvaluate reference based
-> evaluate_nr.py -> evaluate no ref images
-> dataset.py -> load both datasets
-> dataloader -> llve loader 

-> uw_video/ -> sample video dataset
-> data/ -> sample image dataset
-> checkpoints/ -> phishnet checkpoint
-> etc/
/Users/aditya/Downloads/pisces-uv/src/llve/etc/predict_mask.py : masks
/Users/aditya/Downloads/pisces-uv/src/llve/etc/predict_mask_low.py : masks

/Users/aditya/Downloads/pisces-uv/src/llve/etc/CMP/predict_optical_flow.py : of
/Users/aditya/Downloads/pisces-uv/src/llve/etc/CMP/predict_optical_flow_low.py : of

-> metrics/ -> code for metrics 
fastvqa : /Users/aditya/Downloads/pisces-uv/src/llve/metrics/FAST-VQA-and-FasterVQA/vqa.py
niqe : /Users/aditya/Downloads/pisces-uv/src/llve/metrics/NIQE/niqe_metric.m
pcqi : /Users/aditya/Downloads/pisces-uv/src/llve/metrics/PCQI/evaluate.m
uiqm : /Users/aditya/Downloads/pisces-uv/src/llve/metrics/UIQM/demo_UIQM.m
vsfa : /Users/aditya/Downloads/pisces-uv/src/llve/metrics/VSFA/test_demo.py

-> outputs/ -> sample outputs 


slowtv
-> /Users/aditya/Downloads/pisces-uv/src/slowtv/api/quickstart/run.py

## Organisation of the `pisces` package
* `./src` contains the following scripts.
    - `01. train.py` (and equivalently `train_hdr_depth.py` and `train_hdr_depth2.py` which are slight modifications)
    - `02. test.py` (and equivalently `test_hdr_depth.py` and `test_hdr_depth2.py` which are slight modifications)
    - `03. model.py` (and equivalently `model_hdr_depth.py` and `model_hdr_depth2.py` which are slight modifications)
    - `04. dataset.py` to load the dataset for the neural network.
    - `07. load_raw.py` to load images in RAW format.
    - `05. preprocess.py` to preprocess the dataset and remove backscatter.
    - `08. estimate_backscatter.py` to estimate backscatter in an image.
    - `06. metrics.py` constains implementations of the metrics used.

* `./depth` contains the files required to estimate the depth map using Depth Boosting [[2]](http://yaksoy.github.io/highresdepth/).

* `./eval` contains routines to evaluate the model using both reference and non-reference metrics.

* `./ops` and `slice.py` contains utility functions to implement the bilateral grid-based upsampler.

First, prepare your own traning data and put it in the folder `./data`. By default, the code takes input images and ground truth from `./data/train` and `./data/gt` and you can also change the path in `train.py` and `dataloader.py`.

Second, you need to predict plausible optical flow for your ground truth images and put it in the folder `./data/flow`. In our paper, we first perform instance segmentation to get object masks using the opensource toolkit **[detectron2](https://github.com/facebookresearch/detectron2)**. Then we utilize the pretrained **[CMP](https://github.com/XiaohangZhan/conditional-motion-propagation)** model to generate the optical flow we need. 

**Update**:
- [x] The prediction code and example images now are released in `etc` folder.
- [ ] The noise generation code to be updated.

Finally, you can train models on your own data by running
```shell
cd StableLLVE
python train.py 
```
You can replace the U-Net with your own model for low light image enhancement. The model will be saved in the folder `./logs`.

#### Testing
You can put your test images into the folder `./data/test` and just run
```shell
cd StableLLVE
python test.py
```

## Model ##
- [x] checkpoint.pth (This model is trained with the synthetic clean data reported in the paper. It may be unsuitable for noisy data.)


download kbr.ckpt in src/slowtv/api/quickstart

FAST_VQA_3D_1*1.pth
FAST_VQA_B_1*4.pth
to src/llve/metrics/FAST-VQA-and-FasterVQA/pretrained_weights

download CMP checkpoint to src/llve/etc/CMP/experiments/semiauto_annot/resnet50_vip+mpii_liteflow/checkpoints

note : some paths might be broken