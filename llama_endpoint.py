import datetime, os
import torch
import torch.distributed as dist
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from distributed_inference import setup_model_parallel, generate_distributed, generate_distributed_child

os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['NCCL_P2P_DISABLE'] = '1'

DEFAULT_KWARGS = {
    'do_sample': True,
    'top_p': 0.9,
    'temperature': 0.9,
    'epsilon_cutoff': 0,
    'num_beams': 1,
    'num_return_sequences': 1,
    'max_new_tokens': 256,
    'return_dict_in_generate': False, 
    'output_scores': False
}

# After splitting batches among GPUs, this will split into inidividuals runs on each GPU
BATCH_SIZE = 6

MODEL_NAME = str(os.environ.get("MODEL_NAME", 'meta-llama/Llama-2-7b-chat-hf'))
MODEL_PORT = int(os.environ.get("MODEL_PORT", 8474))

app = Flask(__name__)

def setup():
    global tokenizer, model, device, local_rank, world_size, multi_gpu_inference

    multi_gpu_inference = torch.distributed.is_available() # and "--no_dist" not in sys.argv

    # Got this error?: "RuntimeError: CUDA unknown error - this may be due to an incorrectly set up environment, e.g. changing env variable CUDA_VISIBLE_DEVICES after program start. Setting the available devices to be zero."
    # This is a problem with the node itself, CUDA env needs to be reset (requires sudo)

    # Got this error?: "RuntimeError: probability tensor contains either inf, nan or element < 0"
    # This is caused sometimes when loading the model in 16-bit: "torch_dtype=torch.float16"

    device = 'auto'
    if "LOCAL_RANK" in os.environ and multi_gpu_inference:
        local_rank, world_size = setup_model_parallel()
        device = f'cuda:{local_rank}'
        print(f'Loading model to device: {device}')

    print(f'Loading model: {MODEL_NAME}...')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        # bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map=device,
        # torch_dtype=torch.float16,
        load_in_8bit=True,
        rope_scaling={"type": "dynamic", "factor": 2},
        quantization_config=bnb_config
    )
    model.config.pad_token_id = model.config.eos_token_id

def split_list(input_list, batch_size):
    return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]

def llama_batch_generate(input_text, **kwargs):
    if kwargs['do_sample']:
        batches = split_list(input_text, BATCH_SIZE)
    else:
        # For beam search, run each sentence individually, with the batch size
        # used for the different beams
        batches = split_list(input_text, 1)

    print(f"Split input into {len(batches)} batches on {device}")
    generation = []
    for idx, batch in enumerate(batches):
        print(f"Generating batch {idx}/{len(batches)} on {device}")
        torch.cuda.empty_cache()
        generation += llama_generate(batch, **kwargs)

    return generation

def llama_generate(input_text, **kwargs):
    start_time = datetime.datetime.now()

    params = DEFAULT_KWARGS.copy()
    for k, kwarg in kwargs.items():
        if k in params.keys():
            params[k] = kwarg

    if not params['do_sample']:
        del params['top_p']
        del params['temperature']
        del params['epsilon_cutoff']

    if params['do_sample']:
        del params['num_beams']
        del params['num_return_sequences']

    print(f'Generating {len(input_text)} examples on {device} with params {params}')

    inputs = tokenizer(
        input_text, 
        padding=True,
        return_tensors="pt"
    ).to(model.device)
    
    generation = model.generate(
        **inputs, 
        **params,
        use_cache=True
    )
    
    # If 'output_scores': True, return those scores 
    return_scores = not torch.is_tensor(generation)
    if return_scores:
        output = generation.sequences
    else:
        output = generation

    # Delete input text from output so it isn't returned to user
    if params['do_sample']:
        for i in range(inputs['input_ids'].size(0)):
            input_length = inputs['input_ids'][i].size(0)
            output[i, :input_length] = model.config.bos_token_id
    else:
        # NOTE: For beam search we only use one candidate and return n responses
        input_length = inputs['input_ids'][0].size(0)
        output[:, :input_length] = model.config.bos_token_id

    if return_scores:
        if params['do_sample']:
            transition_scores = model.compute_transition_scores(
                generation.sequences, generation.scores, normalize_logits=False
            )
        else:
            transition_scores = model.compute_transition_scores(
                generation.sequences, generation.scores, generation.beam_indices, normalize_logits=False
            )
        transition_scores[~torch.isfinite(transition_scores)] = 0
        seq_prob = torch.sum(transition_scores, axis=1)

        # Beam search will return -log(p(y|x)), sampling return p(y|x)
        if params['do_sample']: seq_prob = -torch.log(seq_prob)
        
        seq_prob = seq_prob.tolist()

    # Measure generation time
    duration = (datetime.datetime.now() - start_time).total_seconds()
    gen_length = output.shape[1] - inputs['input_ids'].shape[1]
    print(f"Generated {gen_length} tokens in {duration:.2f}s at {gen_length/duration:.2f} tok/s on {device}.")

    if output.shape[0] == 1:
        generation = tokenizer.decode(
            output[0], 
            skip_special_tokens=True
        )
        generation = [generation]
    else:
        generation = tokenizer.batch_decode(
            output, 
            skip_special_tokens=True
        )

    if return_scores:
        return [(g, s) for g, s in zip(generation, seq_prob)]
    
    if not params['do_sample']:
        # Package multiple beam outputs into single arrays
        generation = [generation]
            
    return generation

@app.route('/llama', methods=['POST'])
def llama():
    data = request.json

    if not multi_gpu_inference:
        if world_size > 1:
            print(f"Only using 1 GPU to generate, despite having {world_size} available GPUs")
        generation = llama_generate(**data)
        return jsonify({'generation': generation})

    final_generation = generate_distributed(request.json, llama_batch_generate)

    if not data['do_sample']:
        # For beam search: number of inputs != number of outputs
        # Decompress the generation arrays
        final_generation = [i for j in final_generation for i in j]

    response = {'generation': final_generation}
    return jsonify(response)

if __name__ == '__main__':
    setup()

    if multi_gpu_inference:
        try:
            dist.get_rank()
        except RuntimeError as e:
            raise RuntimeError(f'Failed to initialize distributed setup. Did you mean to run `torchrun --nproc_per_node 8 llama_endpoint.py` or `python llama_endpoint.py --no_dist`? Threw: {e}')

    if not multi_gpu_inference or dist.get_rank() == 0:
        print(llama_generate("What is 1+1?")) # Sanity check
        app.run(host='0.0.0.0', port=MODEL_PORT)
    else:
        while True:
            try:
                generate_distributed_child(llama_batch_generate)
            except Exception as e:
                print(f'Device {device} threw exception: {e}')
                pass
