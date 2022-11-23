import gc
import sys
import os
import random
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from eval.evaluate import predict

from nets import deepEM
from loader.prepData import prepdata
from loader.prepNN import prep4nn
from utils import utils
from torch.profiler import profile, record_function, ProfilerActivity

# from memory_profiler import profile

# @profile
def main():
    # read predict config
    # set config path by command line
    inp_args = utils._parsing()
    config_path = getattr(inp_args, 'yaml')
    # config_path = '/home/julio/repos/event_finder/DeepEventMine_fork/experiments/pubmed100/configs/predict-pubmed-100.yaml'


    # set config path manually
    # config_path = 'configs/debug.yaml'

    with open(config_path, 'r') as stream:
        pred_params = utils._ordered_load(stream)

    # Fix seed for reproducibility
    os.environ["PYTHONHASHSEED"] = str(pred_params['seed'])
    random.seed(pred_params['seed'])
    np.random.seed(pred_params['seed'])
    torch.manual_seed(pred_params['seed'])

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load pre-trained parameters
    with open(pred_params['saved_params'], "rb") as f:
        parameters = pickle.load(f)

    parameters['predict'] = True

    # Set predict settings value for params
    parameters['gpu'] = pred_params['gpu']
    parameters['batchsize'] = pred_params['batchsize']
    print('GPU available:' ,torch.cuda.is_available())
    if parameters['gpu'] >= 0:
        device = torch.device("cuda:" + str(parameters['gpu']) if torch.cuda.is_available() else "cpu")
        # torch.cuda.set_device(parameters['gpu'])
    else:
        device = torch.device("cpu")
    parameters['device'] = device

    # Set evaluation settings
    parameters['test_data'] = pred_params['test_data']

    parameters['bert_model'] = pred_params['bert_model']

    result_dir = pred_params['result_dir']
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    parameters['result_dir'] = pred_params['result_dir']

    # raw text
    parameters['raw_text'] = pred_params['raw_text']
    parameters['ner_predict_all'] = pred_params['raw_text']
    parameters['a2_entities'] = pred_params['a2_entities']
    parameters['json_file'] = pred_params['json_file']

    print(' Processing data')
    test_data = prepdata.prep_input_data(
        pred_params['test_data'], parameters, json_file=parameters['json_file'])
    # nntest_data, test_dataloader = read_test_data(test_data, parameters)
    test = prep4nn.data2network(test_data, 'predict', parameters)

    if len(test) == 0:
        raise ValueError("Test set empty.")
    #leak?    
    nntest_data = prep4nn.torch_data_2_network(
        cdata2network=test, params=parameters, do_get_nn_data=True)
    te_data_size = len(nntest_data['nn_data']['ids'])

    test_data_ids = TensorDataset(torch.arange(te_data_size))
    test_sampler = SequentialSampler(test_data_ids)
    test_dataloader = DataLoader(
        test_data_ids, sampler=test_sampler, batch_size=parameters['batchsize'])


    print('Loading mode')
    deepee_model = deepEM.DeepEM(parameters)

    model_path = pred_params['model_path']

    # Load all models
    print('Loading checkpoints mode')
    utils.handle_checkpoints(model=deepee_model,
                             checkpoint_dir=model_path,
                             params={
                                 'device': device
                             },
                             resume=True)

    deepee_model.to(device)

    # with profile(activities=[
    #         ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,with_stack=True) as prof:
    #     with record_function("model_inference"):




    print('predicting')
    predict(model=deepee_model,
            result_dir=result_dir,
            eval_dataloader=test_dataloader,
            eval_data=nntest_data,
            g_entity_ids_=test_data['g_entity_ids_'],
            params=parameters,
            write_files = True)

    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


    #     # Print aggregated stats
    # print(prof.key_averages(group_by_stack_n=5).table(
    #     sort_by="self_cuda_time_total", row_limit=10))

    # prof.export_chrome_trace("trace.json")

# @profile
def read_test_data(test_data, params):

    test = prep4nn.data2network(test_data, 'predict', params)

    if len(test) == 0:
        raise ValueError("Test set empty.")

    nntest_data = prep4nn.torch_data_2_network(cdata2network=test, params=params, do_get_nn_data=True)
    te_data_size = len(nntest_data['nn_data']['ids'])

    test_data_ids = TensorDataset(torch.arange(te_data_size))
    test_sampler = SequentialSampler(test_data_ids)
    test_dataloader = DataLoader(test_data_ids, sampler=test_sampler, batch_size=params['batchsize'])
    return nntest_data, test_dataloader




if __name__ == '__main__':
    main()
