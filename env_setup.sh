#!/bin/bash
# thesis_env_setup.sh

# echo "=== Thesis Experiment Environment Setup ==="

# # 1. Create conda environment
# conda create -n thesis-webagent python=3.10 -y
# conda activate thesis-webagent

# # 2. Install PyTorch with CUDA
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# # 3. Install web agent frameworks
# pip install browsergym
# git clone https://github.com/web-arena-x/webarena.git
# cd webarena && pip install -e . && cd ..

# # 4. Install LLM inference
# pip install transformers accelerate bitsandbytes vllm

# # 5. Install pattern mining
# pip install mlxtend
# # SPMF downloaded separately (Java)

# # 6. Install process mining
# pip install pm4py

# # 7. Install ML/analysis stack
# pip install scikit-learn imbalanced-learn
# pip install pandas numpy matplotlib seaborn
# pip install scipy statsmodels

# # 8. Install experiment tracking (optional but recommended)
# pip install wandb mlflow

# # 9. Install development tools
# pip install pytest black isort jupyter

# # 10. Verify installation
# python -c "
# import torch
# import transformers
# import sklearn
# import pm4py
# print('✓ PyTorch:', torch.__version__, '- CUDA:', torch.cuda.is_available())
# print('✓ Transformers:', transformers.__version__)
# print('✓ Scikit-learn:', sklearn.__version__)
# print('✓ PM4Py:', pm4py.__version__)
# print('All dependencies installed successfully!')
# "

# echo "=== Setup Complete ==="

python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer

models = [
    'meta-llama/Llama-3.2-3B-Instruct',
    'Qwen/Qwen2.5-7B-Instruct',
    'mistralai/Mistral-7B-Instruct-v0.3'
]

for model_id in models:
    print(f'Downloading {model_id}...')
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype='auto',
        device_map='auto'
    )
    print(f'✓ {model_id} downloaded')
"