import os
import torch
import torch.distributed as dist
from flask import Flask, request, jsonify
from comet import download_model, load_from_checkpoint

from distributed_inference import setup_model_parallel, generate_distributed, generate_distributed_child, run_metric

BATCH_SIZE = 100

COMET_PORT = int(os.environ.get("COMET_PORT", 8476))

app = Flask(__name__)

def setup():
    global comet_metric, comet_kiwi_metric, local_rank, multi_gpu_inference

    multi_gpu_inference = torch.distributed.is_available()

    if "LOCAL_RANK" in os.environ and multi_gpu_inference:
        local_rank, _ = setup_model_parallel()
        print(f'Loading model to device: cuda:{local_rank}')

    comet_metric = load_from_checkpoint(download_model("Unbabel/wmt22-unite-da"))
    comet_kiwi_metric = load_from_checkpoint(download_model("Unbabel/wmt22-cometkiwi-da"))


def comet_evaluate(data):
    def metric_func(data):
        evaluation = comet_metric.predict(data, batch_size=BATCH_SIZE, gpus=[local_rank])
        return evaluation.scores
    return run_metric(data, metric_func)


def comet_kiwi_evaluate(data):
    def metric_func(data):
        evaluation = comet_kiwi_metric.predict(data, batch_size=BATCH_SIZE, gpus=[local_rank])
        return evaluation.scores
    return run_metric(data, metric_func)


@app.route('/comet_eval', methods=['POST'])
def comet_eval():
    final_scores = generate_distributed(request.json, comet_evaluate)
    response = {'scores': final_scores}
    return jsonify(response)


@app.route('/comet_kiwi_eval', methods=['POST'])
def comet_kiwi_eval():
    if int(os.environ.get("WORLD_SIZE", -1)) > 1:
        raise NotImplementedError(f"Multi-GPU, reference-free evaluation has not been implemented.")

    final_scores = generate_distributed(request.json, comet_kiwi_evaluate)
    response = {'scores': final_scores}
    return jsonify(response)


if __name__ == '__main__':
    setup()

    if multi_gpu_inference:
        try:
            dist.get_rank()
        except RuntimeError as e:
            raise RuntimeError(f'Failed to initialize distributed setup. Did you mean to run `torchrun --nproc_per_node X {__name__}`? Threw: {e}')

    if not multi_gpu_inference or local_rank == 0:
        data = [
            {
                "src": "Dem Feuer konnte Einhalt geboten werden",
                "mt": "The fire could be stopped",
                "ref": "They were able to control the fire."
            },
            {
                "src": "Schulen und Kindergärten wurden eröffnet.",
                "mt": "Schools and kindergartens were open",
                "ref": "Schools and kindergartens opened"
            }
        ]
        print(comet_evaluate(data)) # Sanity check

        app.run(host='0.0.0.0', port=COMET_PORT)
    else:
        while True:
            try:
                generate_distributed_child(comet_evaluate)
            except Exception as e:
                print(f'Device cuda:{local_rank} threw exception: {e}')
                pass
