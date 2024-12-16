# DLSys_Final

## Dependencies
This project requires [torchtune](https://github.com/YOUR_GITHUB_USERNAME/torchtune), a PyTorch library.

### Installation
1. Clone the torchtune repository:
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/torchtune.git
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
