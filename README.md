## PISCES<sup>uv</sup> : Physically-Informed Subaquatic Color Enhancement System for Underwater Video

![alt text](logo.png "pisces")

## Description
`pisces-uv` contains an implementation of an algorithm to enhance the visual quality of underwater videos using a deep neural network that leverages prior knowledge from a physical model. This approach ensures temporal consistency between the generated frames based on [[1]](https://ieeexplore.ieee.org/document/9578889). The base model achieves real-time processing on high resolution images using a lightweight neural network and a bilateral grid-based upsampler [[2]](https://groups.csail.mit.edu/graphics/hdrnet/), even with limited computing resources. The model is trained using a loss function that combines image reconstruction losses with a colour and smoothness loss to yield optimal results. This is in partial fullfilment for the course _E9 208 : Digital Video - Perception and Algorithms_, August 2023 Term, _IISc Bangalore_. All the code has been written in [Python 3](https://www.python.org).

## Organisation of the `pisces-uv` package
* `./src` contains the following folders and scripts.
    * `ffn`
        - `01. demo.py` to test the FastFlowNet algorithm on two images.
        - `02. run_uw.py` to obtain the flows at resolutions $D_h$=512 and $D_l$=256 for all images in a folder, saved as `.npy` files.
        - `03. visualize_flow.py` to visualize a random flow file using a colorwheel.
    * `llve`
        - `data/` - sample image dataset ($\mathcal{I}$)
        - `uwvideo/` - sample video dataset ($\mathcal{V}$)
        - `checkpoints/` - contains the PhISHNet checkpoint.
        - `etc/`
            - `predict_mask.py` to run Detectron2 on images with size $D_h$.
            - `predict_mask_low.py` to run Detectron2 on images with size $D_l$.
            - `CMP/predict_optical_flow.py` to predict OF on images with size $D_h$.
            - `CMP/predict_optical_flow_low.py` to predict OF on images with size $D_l$.
        - `metrics/`
            - `fastvqa` - use `vqa.py` to obtain the [Fast-VQA and FasterVQA](https://github.com/VQAssessment/FAST-VQA-and-FasterVQA) metrics.
            - `niqe` - use `niqe_metric.m` to obtain the [NIQE](https://in.mathworks.com/help/images/ref/niqe.html) metric.     
            - `pcqi` - use `evaluate.m` to obtain the [PCQI](https://ieeexplore.ieee.org/document/7289355) metric.     
            - `uiqm` - use `demo_UIQM.m` to obtain the [UIQM](https://ieeexplore.ieee.org/document/7305804) metric.     
            - `vsfa` - use `test_demo.py` to obtain the [VSFA](https://github.com/lidq92/VSFA/tree/master) metric.     
        - `outputs/` - contains sample outputs for a video from the VDD-C Dataset.
        - `01. train_PhishNet.py` and `train_PhishVideo.py` are the training scripts.
        - `02. test_PhishVideo.py` to test the model from a `.ckpt` file. Use `--legacy=True` for PhishNet.
        - `03. dataset.py` and `dataloader.py` contain classes detailing both datasets : $\mathcal{I}$ and $\mathcal{V}$.
        - `04. model.py` contains the original LLVE model (for reference, not used).
        - `05. model_depthv2.py` contains the PhISHNet model.
        - `06. utils.py` containing PhISHNet utility files. 
        - `07. load_raw.py` contains loading utilities. 
        - `08. perceptual_losses.py` contains code for $L_{FID}$
        - `09. warp.py` containing the warping utility functions.
        - `10. slice.py` contains utility functions to implement the bilateral grid-based upsampler.
        - `11. metrics.py` contains utilities for the metrics.
        - `12. evaluate.py` to evaluate metrics for images with GT (reference-based).
        - `13. evaluate_nr.py` to evaluate metrics for images without GT (non reference-based).
    * `slowtv`
        - `api/quickstart/run.py` is the script to obtain depthmaps for all the video frames.
        - The rest of the files are helpers/utility scripts.

## Notes

- Prepare your training data and put images from $\mathcal{I}$ in `./src/data` and $\mathcal{V}$ in `./src/uw_video/` in the folder `images`.
- Overall, you would need three different environments, for `ffn`, `llve` and `slowtv` respectively. Follow the official repositories for instructions.
- To predict plausible OF for $\mathcal{I}$, first generate the object masks using instance segmentation from `src/llve/etc/` using [Detectron2](https://github.com/facebookresearch/detectron2) and generate the OF using the pretrained [CMP](https://github.com/XiaohangZhan/conditional-motion-propagation) model in `src/llve/etc/CMP/` by downloading the CMP checkpoint to `src/llve/etc/CMP/experiments/semiauto_annot/resnet50_vip+mpii_liteflow/checkpoints/`.
- To generate the OF for $\mathcal{V}$, use the [FastFlowNet](https://github.com/ltkong218/FastFlowNet) model in `ffn` by downloading the checkpoints `FAST_VQA_3D_1*1.pth` and `FAST_VQA_B_1*4.pth` to `src/llve/metrics/FAST-VQA-and-FasterVQA/pretrained_weights/`.
- To generate depthmaps for $\mathcal{V}$, use the [SlowTV_Monodepth model](https://github.com/jspenmar/slowtv_monodepth/tree/main/src) at `src/slowtv` by downloading the `kbr.ckpt` checkpoint to `src/slowtv/api/quickstart/`.
- Finally train the model with `src/llve/train_PhishNet.py` or `src/llve/train_PhishVideo.py`.
- 

## Authors
* [Aditya C](mailto:adichand20@gmail.com), Department of Electrical and Communication Engineering, IISc Bangalore.
> *For questions or suggestions, please contact: adichand20@gmail.com*

## References
<a id="1">[1]</a> 
F. Zhang, Y. Li, S. You and Y. Fu, "Learning Temporal Consistency for Low Light Video Enhancement from Single Images," 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Nashville, TN, USA, 2021, pp. 4965-4974, doi: 10.1109/CVPR46437.2021.00493.

<a id="2">[2]</a> 
Michaël Gharbi, Jiawen Chen, Jonathan T. Barron, Samuel W. Hasinoff, and Frédo Durand. 2017. Deep bilateral learning for real-time image enhancement. ACM Trans. Graph. 36, 4, Article 118 (August 2017), 12 pages.
