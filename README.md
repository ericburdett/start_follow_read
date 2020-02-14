# Start Follow Read - Additional Instructions
This repository contains an updated version of Start-Follow-Read. It fixes some conda dependencies in the .yaml file and includes some helpful job scripts for running on BYU's Supercomputer. Included as well are some additional instructions to allow for easier use.

## PyTorch Source Code Fix
Unfortunately, there was a change made on a version of CUDNN that made Start-Follow-Read unusable on the supercomputer. Currently, there is a workaround by modifying a few lines of the pytorch source code to not use the broken CUDNN function and instead use a PyTorch implementation.

To get a basic version of Start-Follow-Read up and running, follow these steps:
1. Create the Conda Environment
2. Modify the PyTorch source code found in the Conda Environment you are using
3. Execute the 'run_hwr.py' command
4. Execute the 'run_decode.py' command
5. Access the results

### Create the Conda Environment

If you do not have Anaconda installed yet for your user on the supercomputer, there is a file in the handwriting compute directory that will install Anaconda for you. Run the following script after you have navigated to this file in the compute directory.

```
sh Anaconda3-2019.03-Linux-x86_64.sh
```

If you have already installed Anaconda before but need the conda environment initialized, modify the following command with your username and the path to wherever your installation to Anaconda is found:

```
eval "$(/fslhome/<USERNAME>/anaconda3/bin/conda shell.bash hook)"
```

Create the environment from the dependency file. The .yaml in this forked repository should have all the dependencies up to date:

```
conda env create -f environment.yaml
```

Activate the Start-Follow-Read environment to use the specific Start-Follow-Read dependencies:

```
conda activate sfr_env
```

### Modify the PyTorch Source Code

Navigate to the directory with the PyTorch source code that is contained within the 'sfr_env' anaconda environment. This is usually found in this directory:

```
/fslhome/<USERNAME>/anaconda3/envs/sfr_env/lib/python2.7/site-packages/torch
```

We now need to modify the source code to force PyTorch to use its own implementation of the Grid_Sample method, rather than the CUDNN version.

Navigate to the following file within the torch direcory listed above:

```
nn/_functions/vision.py
```

Comment out lines 27-34 and 46-59 and correct the indentation issues. We are simply removing the code contained in the body of the IF statements in the GridSample forward and backward functions.

### Run the Code

Navigate to the Start-Follow-Read directory and open the job.sh file. This contains the code necessary to run a basic job on BYU's Supercomputer.

Modify the paths to correspond to your username and the source to your input images. It should look something like this:
```
eval "$(/fslhome/burdett1/anaconda3/bin/conda shell.bash hook)"

conda activate sfr_env

python run_hwr.py IMG_SOURCE_DIRECTORY sample_config_60.yaml IMG_DESTINATION_DIRECTORY

python run_decode.py sample_config_60.yaml IMG_DESTINATION_DIRECTORY
```

Note that the rest of the information in the job.sh file contains configuration settings for running on the Supercomputer. If you'd like more walltime, more nodes, gpu's, etc., you need to modify this file. Currently, the job.sh file will place the job in the test queue so that execution will begin almost immediately. However, bigger jobs will need to be placed in the normal queue.

Submit the job to the supercomputer:
```
sbatch job.sh
```

Output for the job will be placed in a file called slurm-<JOB_NUMBER>.out.

If you'd like to see the current status for the job. Run the following:

```
scontrol show job <JOB_NUMBER>
```

Note that the job number should be provided for you when you submit the job.

### Observe the Results

Results should be available in the IMG_DESTINATION_DIRECTORY that you specified in the commands above.

There should be 3 files image in this directory.
* *.npz - The output files created from the run_hwr command
* *.txt - Transcription of the image line by line
* *.png - Visualization of what was produced from the start of line and line follower modules. Numbers here correspond to the line numbers in the .txt file.

# Start Follow Read - Original Readme

This repository is the implementation of the methods described in our paper [Start, Follow, Read: Full-Page End-to-end Handwriting Recognition](http://openaccess.thecvf.com/content_ECCV_2018/html/Curtis_Wigington_Start_Follow_Read_ECCV_2018_paper.html).
All steps to reproduce our results for the [ICDAR2017 Competition on Handwritten Text Recognition on the READ Dataset](https://scriptnet.iit.demokritos.gr/competitions/8/) can be found in this repo.

This code is free for academic and research use. For commercial use of our code and methods please contact [BYU Tech Transfer](techtransfer.byu.edu).

We will also include pretrained models, results, and the segmentation data inferred during training. These can be found on the [release page](https://github.com/cwig/start_follow_read/releases).


## Dependencies

The dependencies are all found in `environment.yaml`. They are installed as follows.
```
conda env create -f environment.yaml
```

The environment is activated as `source activate sfr_env`.

You will need to install the following libraries from source. warp-ctc is needed for training.
PyKaldi is used for the language model. A pretrained Start, Follow, Read network can run
without either.
- [warp-ctc](https://github.com/SeanNaren/warp-ctc)
- [PyKaldi](https://github.com/pykaldi/pykaldi)

## Prepare Data

Download Train-A and Train-B from the competition [website](https://scriptnet.iit.demokritos.gr/competitions/8/). You need `Train-A.tbz2`, `Train-B_batch1.tbz2`, `Train-B_batch2.tbz2`. Put them in the data folder. You will also need `Test-B2.tgz` if you plan on submitting results to the competition website.

#### Extract Files

```
mkdir data
cd data
tar jxf Train-A.tbz2
tar jxf Train-B_batch1.tbz2
tar jxf Train-B_batch2.tbz2
cd ..
```

#### Prepare Train-A

This process can be a bit slow because the normalization code is inefficient.
This extracts start-of-line positions, line follower targets, and normalized line images.

```
python preprocessing/prep_train_a.py data/Train-A/page data/Train-A data/train_a data/train_a_training_set.json data/train_a_validation_set.json  
```

#### Prepare Train-B

This extracts only the GT lines from the XML.

```
python preprocessing/prep_train_b.py data/Train-B data/Train-B data/train_b data/train_b_training_set.json data/train_b_validation_set.json
```

#### Prepare Test data

Currently we only support running the tests for the Test-B task, not Test-A. When we compute the results for the Test-B while fully exploiting the competition provided regions-of-interest (ROI) we have to do a preprocessing step. This process masks out parts of the image that are not contained in the ROI.

```
python preprocessing/prep_test_b_with_regions.py data/Test-B data/Test-B data/test_b_roi
```

#### Generate Character Settings

This will generate a character set based on the lines in both Train-A and Train-B.
There should 196 unique characters.
This means the network will output 197 characters to include the CTC blank character.

```
python utils/character_set.py data/train_a_training_set.json data/train_a_validation_set.json data/train_b_training_set.json data/train_b_validation_set.json data/char_set.json
```


## Pretraining

In this example training is performed using a 32 pixel tall images.
I would recommend training on 32 pixel tall images.
Then training the line-level HWR network is retrained afterwards at a high resolution.
The 32 pixel network trains faster and is good enough for the alignment.


All three networks can fit on a 12 GB GPU for pretraining.
Sorry, no graphs of the training and validation loss at this time. Each network will stop training after 10 epochs without any improvement.

A sample SLURM file to pretrain can be found in `slurm_examples/pretrain.sh`. The individual commands for each network are given below.

#### Start of Line

You should expect to be done when the validation loss is around 50-60.

```
python sol_pretraining.py sample_config.yaml  
```

#### Line Follower

You should expect to be done when the validation loss is around 40-50.

```
python lf_pretraining.py sample_config.yaml  
```

#### Handwriting Recognition

You should expect to be done when the validation CER is around 0.50 to 0.55.

```
python hw_pretraining.py sample_config.yaml  
```

#### Copy Weights

After pretraining you need to copy the initial weights into the `best_overall`, `best_validation`, and `current` folders.

```
cp -r data/snapshots/init data/snapshots/best_overall
cp -r data/snapshots/init data/snapshots/best_validation
cp -r data/snapshots/init data/snapshots/current
```

## Training

Training of each component and alignment can be performed independently.
I have run using 4 GPUs.
You could do it on a single GPU but you would have to adapt the code to do that.

For BYU's super computer I run the following SLURM files for 4 GPUs.
You can run the python files independent of the SLURM scripts.

#### Initial Alignment

Before you can train, you have to first run the alignment so there are targets for the validation and the training set.
It will perform alignment over the validation set and the first training group (2000 images total)

A sample SLURM file can be found in `slurm_examples/init_alignment.sh`.

```
python continuous_validation.py sample_config.yaml init

```

#### Training

All of the following are designed to be run concurrently on 4 GPUs. They could be modified to run sequentially, but this would slow training time. If you more GPUs `continuous_validation.py` can be set to run over specific subsets of the dataset so more validation can happen in parallel. We did our experiments using 4 GPUs.

A sample SLURM file can be found in `slurm_examples/training.sh`.

```
CUDA_VISIBLE_DEVICES=0 python continuous_validation.py sample_config.yaml
CUDA_VISIBLE_DEVICES=1 python continuous_sol_training.py sample_config.yaml
CUDA_VISIBLE_DEVICES=2 python continuous_lf_training.py sample_config.yaml
CUDA_VISIBLE_DEVICES=3 python continuous_hw_training.py sample_config.yaml
```

## Retraining

Because we trained the handwriting recognition network at a lower resolution, we need to retrain it. First, we need to segment our line-level images at a high resolution.

A sample SLURM file can be found in `slurm_examples/resegment.sh`.

```
python resegment_images.py sample_config_60.yaml

```

After segmenting we need to retrain the network. We can just use the pretraining code to this.

A sample SLURM file can be found in `slurm_examples/retrain_hwr.sh`.

```
python hw_pretraining.py sample_config_60.yaml  
```

## Validation (Competition)

This section covers reproducing the results for the competition data specifically. The next section will explain it more generally.



#### With using the competition regions-of-interest.

```
python run_hwr.py data/test_b_roi sample_config_60.yaml data/test_b_roi_results
```

```
python run_decode.py sample_config_60.yaml data/test_b_roi_results --in_xml_folder data/Test-B --out_xml_folder data/test_b_roi_xml --roi --aug --lm
```

#### Without using the competition regions-of-interest

```
python run_hwr.py data/test_b sample_config_60.yaml data/test_b_results
```

```
python run_decode.py sample_config_60.yaml data/test_b_results --in_xml_folder data/Test-B --out_xml_folder data/test_b_xml --aug --lm
```

#### Submission

The xml folder needs to be compressed to a `.tar` and then can be submitted to the [online evaluation system](https://scriptnet.iit.demokritos.gr/competitions/8/).

We also include the xml files from our system so you can compute the error with regards those predictions instead of submitting to the online system. This will give you a rough idea of how your results compare to other results. The error rate is not computed the same as on the evaluation server.

``
python compare_results.py <path/to/xml1> <path/to/xml2>
``

## Validation (General)

The network can be run on a collection of images as follows. This process produces intermediate results. The post processing to these results are applied in a separate script.

```
python run_hwr.py <path/to/image/directory> sample_config_60.yaml <path/to/output/directory>
```

The postprocessing has number of different parameters. The most simple version is as follows. The `<path/to/output/directory>` is the same path in the previous command. It will save a text file with the transcription and an image to visualize the segmentation.

```
python run_decode.py sample_config_60.yaml <path/to/output/directory>
```

Run it with test-side augmentation:

```
python run_decode.py sample_config_60.yaml <path/to/output/directory> --aug
```

Run it with the language model:

```
python run_decode.py sample_config_60.yaml <path/to/output/directory> --lm
```

Run it with both the language model and test-side augmentation:

```
python run_decode.py sample_config_60.yaml <path/to/output/directory> --aug --lm
```
