Steps and code to reproduce the work "Supervised Representation Learning Approach towards 
Generalizable Assembly State Recognition"


File overview: 
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


Set-up:

0. Install the public IndustReal dataset: https://github.com/TimSchoonbeek/IndustReal following
   the instructions and directory structure provided
1. Install required packages: pip install -r requirements.txt
2. To use pre-trained weights, download safetensors files from HuggingFace 
   and place them in ./models dir. The results reported in the paper use the weights 
   "resnet34.a1_in1k" and "vit_small_patch16_224.augreg_in21k_ft_in1k"
3. Optional: use error_labels.json (the new labels published together with the manuscript) to
   create the IndustReal error images subset. The image names contain all required information:
   the IndustReal recording and frame name of each image, the error category, the user-intended 
   state, the bounding box, and whether the assembly state is not occluded (indicated with binary
   label 'clean').


Reproducing results:

0. Run train.py, pointing to the directory where the IndustReal data is stored. The default 
   configuration consists of the hyperparameters as used for the experiments reported.
1. Run test.py to test the performance on Industreal Test and the additional synthetic 
   generalization set
2. Run test_errors.py to evaluate the performance on the new IndustReal errors subset.
3. Optional: train.sh provides code to perform steps 0-2 for 5 different seeds 
