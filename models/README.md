# Models

## Original scores
The results presented in the paper are implemented with GPT-2.

The script `./gpt2/train_gpt2.sh` contains the command to run to finetune a GPT-2 model on each dataset.
This code uses `transformers=4.21.2`, `pytorch=1.12.1`, and `pytorch-lighning=1.7.4`.


## Validation
Validation is conducted with Gemma 2B and Phi1.5, using the LLaMA-Factory framework. 

To train these models with LLaMA-Factory, follow the installation instructions on https://github.com/hiyouga/LLaMA-Factory. Let `$LF_DIR` be the directory to the repo.
Then, 
1. Prepare the each dataset for the `pt` mode. Add the prepared datasets to the `$LF_DIR/data/` folder and update `dataset_info.json`.
2. Create config files for training and inference (see `./llama-factory/train/` and `./llama-factory/inference` for examples).
3. Update the `./llama-factory/train_models.sh` script with the model you wish to train.
4. Run `./llama-factory/train_models.sh`.

