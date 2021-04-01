# X2T: Training an X-to-Text Typing Interface with Online Learning from User Feedback

This repository contains code for X2T applied to the gaze and handwriting domains described in the 
[paper](https://openreview.net/pdf?id=LiX3ECzDPHZ). This includes both a real gaze interface that can be operated on a
computer with an available webcam, simulated offline gaze (by replaying data collected in the gaze user study
for the paper), and simulated handwriting.

## Setup Instructions
1. Run `conda create -n YOUR_ENV_NAME python=3.7` to create a new conda env for this project and activate it.
2. Install the package requirements using `pip install -r requirements.txt`.
3. Install the local bci-typing package `pip install -e .`.
4. Install the gym_bci_typing environment using `pip install -e gym-bci-typing/`.
5. Make sure that CUDA 10.0 and cuDNN 7.6.5 for CUDA 10.0 are both installed if you want GPU support.
6. If running into issues with dlib (needed for the real gaze interface with a webcam), 
   you may need to install dependencies. Instructions can be found
[here](https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/).

The following steps are needed to use the included Transformer-XL language model
(for use with the simulated handwriting domain).
1. Run `transformer_xl/tf/sota/download.sh` to download the pretrained Transformer-XL model
on One Billion Words.
2. Run `transformer_xl/tf/fix_cached_data.py` to fix the pretrained model's data cache to
work with the project's directory structure. Alternatively, after running the above download script, replace the folder
   `transformer_xl/pretrained_xl/tf_lm1b/data` with the included
   `transformer_xl/pretrained_xl/tf_lm1b/data_fixed` folder (renaming the folder to `data` as well).


## Running Experiments
The `train.py` script is used for all experiment types. Run `python train.py --help` for information on
all command line arguments.

Each experiment is logged to `experiments/USER_TYPE/SAVE_NAME`, where `USER_TYPE` is the user type of the
experiment (e.g. `real_gaze`) and `SAVE_NAME` is specified by the flag `-s SAVE_NAME`. By default,
`SAVE_NAME` is the date and time of the experiment (e.g. `2020-08-21--18-20-27`). This folder
may contain each of the following, when appropriate for the parameters of the experiment:
- `log_1` contains the TensorBoard training log for adaptive sessions.
- `offline_log_1` contains the TensorBoard offline training log.
- `data.hdf5` is the data for each step recorded during the session (can be used by `-dp` or `-gp` for 
  future sessions). It contains 4 keys ('obses', 'actions', 'rewards', 'targets').
  - 'obses' is a (timesteps, `n_samples`, input dim) shaped array containing the inputs provided by the user. In the
    data from the gaze study, `n_samples` is the default value of 10. `n_samples` is always 1 in the simulated
    handwriting domain.
  - 'actions' is a (timesteps,) shaped array, containing the action selected by the interface at each timestep.
  - 'rewards' is a (timesteps,) shaped array, containing 0 if the user backspaced the action for that timestep, and 1
    otherwise.
  - 'targets' is a (timesteps,) shaped array, containing the true action indices that the user intended to select each
    timestep.
- `calibration_data.hdf5` is the data collected by real gaze users to calibrate default interfaces
  (can be used by `-gp` for future sessions). It contains 2 keys ('signals', 'gaze_labels').
  - 'signals' is a (dataset size, input dim) shaped array containing the gaze feature inputs provided by the user
    during the calibration phase.
  - 'gaze_labels' is a (dataset size, 2) shaped array containing the normalized 2D coordinates of the  
    locations that the user was directed to look at during the calibration phase. It is aligned with 'signals', such
    that each index in the first dimension represents an (input, coordinate) pair for supervised learning.
  - This file also contains an attributes object (.attrs), which contains the 'cam_coord', 'height', 'width',
    'window_height', and 'window_width' of the user's setup during the session calibrated by this data.
- `final_model.zip` contains the final model weights (can be passed to `-l` for future sessions).
- `offline_model.zip` contains the model weights after offline pretraining. (can be passed to `-l` for 
  future sessions).
- `metrics.txt` contains the overall prediction accuracy of the session.
- `params.json` contains the parameters of the session.
- `baseline_estimates.pkl` contains the 2D normalized gaze position estimates of the baseline default interface. It
  is a `n_actions` length list (default 8), and each element is a list of normalized 2D estimates that the default
  interface predicted for that action.

Also, if running in baseline mode, the `data.hdf5` file will be copied to
`experiments/USER_TYPE/recent/baseline_data.hdf5`, which can be loaded from in the future using `-ur`.

## Replicating X2T Paper Experiments

### Real Webcam Gaze
1. Run `python train.py -u rg -m b -n N_STEPS` to run the default interface that uses a calibrated nearest-action
   baseline for the desired number of steps. The interface will first run through a calibration procedure where you are
   directed to look at various positions on the screen. Then the typing interface will start. This command was used in
   gaze user study for both collecting data to pretrain the learned reward model in the adaptive sessions, and running
   default interface sessions to compare with the adaptive sessions.

2. Run `python train.py -u rg -ur -n N_STEPS` to pretrain the learned reward model of the interface using
   the offline default interface data collected in the previous step, and then start an adaptive session for the desired
   number of steps, using the mixed learned + default policy. The interface will first run through a calibration
   procedure where you are directed to look at various positions on the screen. Then the typing interface will start.
   
#### Additional Flags
For using the calibrated default baseline with real gaze (any `-m` flag except `l`), the following arguments
should also be set according to the user's webcam and screen setup:
- `-cx CAM_X` the horizontal displacement (cm) of the webcam from the left end of the interface window.
Positive values indicate rightward displacement.
- `-cy CAM_Y` the vertical displacement (cm) of the webcam from the top end of the interface window. Positive values
indicate downward displacement.
- `-ww WINDOW_WIDTH` The width (cm) of the interface window.
- `-wh WINDOW_HEIGHT` The height (cm) of the interface window.

The default values are for a 13-inch MacBook Pro in fullscreen.
   
#### How to Use the Interface:
If using in fullscreen mode (default), make sure your computer only has 1 monitor. Try to keep your face in the relative
center of your webcam, as if your face is outside of the center square region of your webcam (the webcam recording is
horizontally truncated to match the height of the recording), it may not be detected. After the interface has loaded,
there will be `n_actions` different numbers, arranged in a circle. Press `SPACE` to begin the calibration phase.
The number 0 will be colored orange, and you should stare at it until the next number is colored orange. This process
will repeat for all numbers, for a total of two complete cycles. If the orange number is not changing after a
short period of time, it is because your face is not being detected, as the interface will only continue to the next
number once enough data for that number is recorded.

Once calibration is complete, you will be presented with a sentence to type. The top of the screen will display this
sentence, as well what has been typed so far, and the next word in the sentence to type. There will be `n_actions`
different words arranged in a circle, in the same positions as the calibration points. One of these words will always
be the next word to type. Once you have located this word, you should stare at it, then press `SPACE` to begin
recording. After a short period of time, the interface will predict which word you intended to type, and add it to what
has been typed so far. If the interface made an error, the newly typed word will appear red, and you should press 
`BACKSPACE` to undo this word and try again. Otherwise, you should continue to try typing the next word in the
same manner as before. After typing the number of words in the target sentence, the interface will reset with a new 
sentence. The interface will close automatically when `N_STEPS` timesteps have passed (each word prediction by 
the model is a timestep).

You can only undo the most recent word, i.e. you cannot undo then undo again without trying to type another word. You
can only undo a word before you press `SPACE` to start typing the next word. You cannot undo the last word in a
sentence (it will always move on to the next sentence, regardless if it was correct or not). Press the `ESC` key at
any time to terminate the program early. Press the `p` key during a step to pause the program. Press `p` while
paused to unpause the program. After unpausing, any input collected during the previous interrupted step is discarded.
No other keys are used.

### Simulated Offline Gaze
All data from the gaze user study described in the paper is found in the folder `experiments/gaze_study`. The folder
`offline` contains data used to pretrain the adaptive interfaces. The folder `default` contains data for the
default interface sessions. The folder `x2t` contains data for the adaptive interface sessions. Each of these
folders contains folders numbered 0-11, one for each of the 12 participants in the study. Each of these
numbered folders contains the `calibration_data.hdf5` and `data.hdf5` associated with those sessions. The
`x2t` folders also contain `offline_model.zip`, which are the weights for the models that were pretrained and
used to initialize the adaptive sessions in the study and `baseline_estimates.pkl`, which contains the default
interface baseline estimates on the data collected during the adaptive sessions in the study.

The `-gp` flag should always be set to a folder containing a `data.hdf5` file produced by real gaze session.
For using the calibrated default baseline with simulated gaze (any `-m` flag except `l`), the folder should also
contain a `calibration_data.hdf5` file produced by a real gaze session. Here is an example of a command to run a
simulated offline gaze default interface session, using data from participant 0
during their adaptive session:

`python train.py -u og -m b -gp experiments/gaze_study/x2t/0/`

For simulated adaptive sessions, you can pretrain the learned reward model by setting the `-dp` flag to a 
`data.hdf5` file produced by a session. For example, to pretrain using the data from participant 0 that was
collected for this purpose in the study, you should add the flag `-dp experiments/gaze_study/offline/0/data.hdf5`.
You can also initialize the learned reward model weights directly using the `-l` flag. For example, to initialize
using the same pretrained initialization used for the adaptive session of participant 0, you should add the flag
`-l experiments/gaze_study/x2t/0/offline_model.zip`. 

#### Coadaptation Experiments
To recreate the coadaptation experiments in the appendix of the paper, run `./scripts/coadapt.sh`. This script
initializes learned reward models using corresponding pretrained initializations from the adaptive sessions in the
study. The results will be saved in `experiments/offline_gaze/x2t_on_default_data/USER_INDEX` and
`experiments/offline_gaze/default_on_x2t_data/USER_INDEX`, where `USER_INDEX` is a number 0-11, representing
which user from the study the data came from.

#### Reward Noise Experiments
To recreate the reward noise sensitivity experiments in the appendix of the paper, run `./scripts/rew_noise.sh`. 
This script initializes learned reward models using corresponding pretrained initializations from the adaptive sessions
in the study. The results will be saved in `experiments/offline_gaze/rew_noise/NOISE_LEVEL/USER_INDEX`, where
`NOISE_LEVEL` is the fraction the user's rewards that are incorrect (e.g. 0.1), and `USER_INDEX` is a number
0-11, representing which user from the study the data came from.

### Simulated Handwriting 

#### Ablation Experiments
To recreate the full simulated handwriting ablation experiments in the paper, run 
`./scripts/ablation.sh`. The results will be saved to `experiments/sim_uji/ablation/CONDITION/USER_INDEX`,
where `CONDITION` indicates the ablation condition (e.g. default, x2t, x2t_no_pretrain), and `USER_INDEX` is
a number 0-59 representing which writer in the UJI Pen Characters v2 dataset was used.

#### Personalization Experiments
To recreate the full simulated handwriting personalization experiments in the paper, first recreate the ablation
experiments as above, then run `./scripts/personalization.sh`. The results will be saved to
`experiments/sim_uji/personalization/USER_TRAIN/USER_EVAL`, where `USER_TRAIN` is the writer in the
UJI Pen Characters v2 dataset that the model was trained on, and `USER_EVAL` is the writer that the model was
evaluated on.

#### Handwriting Data Format
The file `online_handwriting.pkl` contains data representing pen strokes from the UJI Pen Characters v2
dataset. It is a dictionary with 27 keys, one for each lower case letter and space, e.g. 'a', ' '. Each key indexes
into a length 120 list, 2 elements for each writer in the dataset. They are ordered as (writer 0, writer 0, writer 1,
writer 1, ...). Each element of this list is a drawing of a character, and is represented list of arrays. Each of these
arrays represents a pen stroke, and is of the shape (None, 2), where the first dimension represents the different
positions in the stroke (in order), and the second dimension is the coordinate of each position. Each of the positions
in a stroke should be connected in order by lines, and all the strokes together form a character. There should not be
additional lines connecting the strokes in a character.

### Hyperparameters
The default hyperparameters in `train.py`, which is used by all the above scripts, are those specified in the
X2T paper. However, it is recommended to instead use a learning rate of `5e-4` in the gaze domain
(both real and simulated) to have more stable performance.

## Notebooks
See `notebooks/plots.ipynb` to create the plots in the paper, and `notebooks/anova.ipynb` to obtain the ANOVA
results in the paper.

## Citation
If you find this repository useful for your work, please cite:
```
@inproceedings{
gao2021xt,
title={X2T: Training an X-to-Text Typing Interface with Online Learning from User Feedback},
author={Jensen Gao and Siddharth Reddy and Glen Berseth and Nicholas Hardy and Nikhilesh Natraj and Karunesh Ganguly and Anca Dragan and Sergey Levine},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=LiX3ECzDPHZ}
}
```

## Acknowledgements
Contains code, models, and data from the PyTorch implementation of "Eye Tracking for Everyone,"
found at https://github.com/CSAILVision/GazeCapture/tree/master/pytorch.
> Kyle Krafka, Aditya Khosla, Petr Kellnhofer, Harini Kannan, Suchi Bhandarkar, Wojciech Matusik and Antonio Torralba.
> “Eye Tracking for Everyone”. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

Contains code and models from "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context," found at
https://github.com/kimiyoung/transformer-xl.
> Dai, Zihang, et al. "Transformer-xl: Attentive language models beyond a fixed-length context."
> arXiv preprint arXiv:1901.02860 (2019).

## Contact
For any questions, bugs, or suggestions, please feel free to contact jenseng@berkeley.edu.
