# DLSys_Final

## Dependencies
This project requires [torchtune](https://github.com/pytorch/torchtune), a PyTorch library.
## Directory Structure
```bash
DLSys_Final/
├── README.md
├── environment.yml
├── Metric_cn.py
├── after_sweep.py
├── models/
│   ├── meta-llama3/
│   ├── Mistral/
│   ├── 
├── torchtune/
│   ├── recipes/
│   │   ├── configs/
│   │   │   ├── llama3/
│   │   │   │   ├── 8B_lora_single_device.yaml
│   │   │   ├── ...
│   │   ├── generate.py
│   │   ├── generate_many.py
│   │   ├── lora_finetune_single_device.py
│   │   ├── ...
```

## Installation
1. Clone the torchtune repository:
```bash
git clone https://github.com/roxqtang/DLSys_Final
```
2. Install the conda environment:
```bash
conda env create --file environment.yml
```
3. Install the torchtune library:
```bash
pip install torchtune torchao huggingface_hub
```
4. Download the llama3 model (you need to get access to the llama3 model from the huggingface website):
```bash
tune download meta-llama/Meta-Llama-3-8B-Instruct \
    --output-dir your_output_dir \
    --hf-token your_huggingface_token
```

## Finetuning Process:
1. set config file:
for Llama3 8B:
in the file
```bash
torchtune/recipes/configs/llama3/8B_lora_single_device.yaml
```
5 Run finetuning:
```bash
tune run lora_finetune_single_device --config llama3/8B_lora_single_device
```
## Evaluation
## Evaluating LLAMA3 8B Instruct finetuned model
For evaluation, we have manually changed the generate.py in torchtune/recipes/generate.py to generate_many.py in order to get multiple generations.
But to evaluate the model, we are using the Metric_cn.py
1. set config file:
in the file
```bash
torchtune/recipes/configs/llama3/8B_lora_single_device.yaml
```
