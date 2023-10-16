import requests
from json import JSONDecodeError

DEFAULT_KWARGS = {
    'do_sample': True,
    'top_p': 0.9,
    'temperature': 0.9,
    'max_new_tokens': 256,
    'num_beams': 1,
    'num_return_sequences': 1,
    'return_dict_in_generate': False, 
    'output_scores': False
}

LLAMA_ENDPOINT = 'http://localhost:{port}/llama'

def generate_llama(prompt, sentence_level=False, port=8474, **kwargs):
    """
    Use an API endpoint to generate with LLaMA.
    """
    params = DEFAULT_KWARGS.copy()
    for k, kwarg in kwargs.items():
        if k in params.keys():
            params[k] = kwarg

    data = { 
        'input_text': prompt
    }
    data.update(params)

    if sentence_level:
        data.update({'stopping_criteria': sentence_level})

    try:
        response = requests.post(LLAMA_ENDPOINT.format(port=port), json=data)
        response_json = response.json()
    except JSONDecodeError:
        raise RuntimeError(f"Llama endpoint failed to respond. Returned: {response}")

    return response_json['generation']

if __name__ == '__main__':
    prompt = "Write a paragraph about animals native to South America."
    generation = generate_llama(prompt)
    print(f'Generation: {generation}')