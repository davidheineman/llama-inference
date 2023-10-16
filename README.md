Host a LLaMA model with an API endpoint using Flask. 

## Setup

Install dependencies:
```sh
pip install -r requirements.txt
```

Run the remote server for **LLaMA** (e.g., on 8 GPUs):
```sh
# Note: NCCL_P2P_DISABLE will allow running multiple NCCL jobs simultaneously
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 --master_port 29500 llama_endpoint.py

# Use environment args to specify custom model name or API port
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MODEL_NAME=meta-llama/Llama-2-70b-chat-hf MODEL_PORT=8474 torchrun --nproc_per_node 8 --master_port 29500 llama_endpoint.py
```

(Optional) You can use this code for distributed inference of any model. For an example of distribed evaluation using COMET, use this:
```sh
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port 29501 comet_endpoint.py
```

(Optional) To query the endpoint from your local computer, you can open an SSH tunnel:
```sh
ssh -N -f -L localhost:8000:localhost:8474 USER_NAME@NODE_NAME.cc.gatech.edu
```

## How to use
When running, you can query the endpoint using `requests`:
```python
import requests
from json import JSONDecodeError

PORT = 8474
LLAMA_ENDPOINT = 'http://localhost:{port}/llama'

prompt = "Write a paragraph about animals native to South America."
data = { 
    'input_text': prompt
}

try:
    response = requests.post(LLAMA_ENDPOINT.format(port=port), json=data).json()
except JSONDecodeError:
    raise RuntimeError(f"Llama endpoint failed to respond. Returned: {response}")

generation = response_json['generation']

print(f'Generation: {generation}')
```

Example generation using the endpoint:
```sh
python generate.py
```