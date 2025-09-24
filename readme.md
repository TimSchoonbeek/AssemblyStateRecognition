Steps and code to reproduce the work "Supervised Representation Learning Approach towards Generalizable Assembly State Recognition"

## Check out the [project page](https://timschoonbeek.github.io/state_rec)

## File overview: 
- train.py contains the script required to perform training with the representation learning 
  framework.
- test.py evaluates the performance of a trained model on the IndustReal test set and the 
  unseen assembly states. The unseen assembly states data is not 
  provided as supplementary materials given the size limitations, but will be publicly hosted.
- test_errors.py evaluates the performance of a trained model on the IndustReal error images.
  Note that to run this test, the new labels (provided in error_labels.json) are required.
- losses.py contain the batch hard triplet and supervised contrastive loss functions with the
  novel ISIL modification, proposed in the manuscript. The batch all triplet loss is also 
  implemented with ISIL but not reported, given the journal space limitations and the general
  outperformance of batch hard compared to batch all.
- models.py contains the implementation of the neural network architectures. The encoders
  are based on .safetensors files of pre-trained weights from Huggingface. The projection 
  head is implemented as outlined in the manuscript.
- datasets.py contains the dataset classes used in train.py, test.py, and test_errors.py
- utils.py contains generic utilization functions
- asr_performance_figure.py recreates the all performance figures of the manuscript, with the 
  IndustReal test, generalization , and error performance.
- error_labels.json contains the new annotations for the error states, including intended state
  and error category 
- train.sh contains the code to execute 5 cycles with different seeds for a given setting. The
  default settings are for the best-performing model: SupCon with ISIL for the ViT-S model.

## Data set-up
The easiest method to set up the data correctly is to download all files from the data repository
connected to the publication. You can find this on the [project page](https://timschoonbeek.github.io/state_rec)
which hosts the most up-to-date data links.
To use the data, unpack all of the tar files into a new directory, such that:
```
data/
├── anchors/
│   ├── images/
│   └── labels.json
├── assembly_state_errors/
│   ├── images/
│   └── labels.json
├── synth/
│   ├── images/
│   └── labels.json
├── train/
│   ├── images/
│   └── labels.json
├── test/
│   ├── images/
│   └── labels.json
└── val/
    ├── images/
    └── labels.json
```

Additionally, the annotations in assembly_states_errors contain our new labels:
- "clean": a Boolean indicating whether the frame demonstrates the entire assembly state,
- "intended": the user-inteded assembly state (without the errors) as defined in IndustReal,
- "error_cat": the error category as defined in our paper. Importantly, we define: 1. Missing
  component, 2. Incorrect component placement, 3. Incorrect component orientation, 4. Part-level error.

Please note that the error category above correspond directly to Fig. 7, but their roman numerical 
does not refer to the integer provided in "error_cat". Hence it is important to follow the definition for 
provided above.

To start training and testing with the data, simply unzip and point to the directories in the 
provided .sh scripts


## Set-up:

0. (optional if the .zips we provide are used) Install the public
   [IndustReal dataset](https://timschoonbeek.github.io/industreal) following
   the instructions and directory structure provided
2. Install required packages: pip install -r requirements.txt
3. To use pre-trained weights, download safetensors files from HuggingFace 
   and place them in ./models dir. The results reported in the paper use the weights 
   "resnet34.a1_in1k" and "vit_small_patch16_224.augreg_in21k_ft_in1k"
4. Optional: use error_labels.json (the new labels published together with the manuscript) to
   create the IndustReal error images subset. The image names contain all required information:
   the IndustReal recording and frame name of each image, the error category, the user-intended 
   state, the bounding box, and whether the assembly state is not occluded (indicated with binary
   label 'clean').


## Reproducing results:

0. Run train.py, pointing to the directory where the IndustReal data is stored. The default 
   configuration consists of the hyperparameters as used for the experiments reported.
1. Run test.py to test the performance on Industreal Test and the additional synthetic 
   generalization set
2. Run test_errors.py to evaluate the performance on the new IndustReal errors subset.
3. Optional: train.sh provides code to perform steps 0-2 for 5 different seeds 

