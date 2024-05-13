import logging
import os
import json
import ConfigSpace as CS
import numpy as np
import argparse
from functools import partial
import math

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant
from sklearn.model_selection import StratifiedKFold

from smac.configspace import ConfigurationSpace
from smac.facade.smac_bohb_facade import BOHB4HPO
from smac.scenario.scenario import Scenario

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchsummary import summary
from torchvision.datasets import ImageFolder

from cnn import torchModel

constraint_model_size = 0
constraint_precision = 0
use_teset_data = False


# Target Algorithm
# The signature of the function determines what arguments are passed to it
# i.e., budget is passed to the target algorithm if it is present in the signature
def cnn_enhanced(cfg, seed, instance, budget, run='1',
                 data_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'micro17flower')):
    """
        Creates an instance of the torch_model and fits the given data on it.
        This is the function-call we try to optimize. Chosen values are stored in
        the configuration (cfg).

        Parameters
        ----------
        cfg: Configuration (basically a dictionary)
            configuration chosen by smac
        seed: int or RandomState
            used to initialize the rf's random generator
        instance: str
            used to represent the instance to use (just a placeholder for this example)
        budget: float
            used to set max iterations for the MLP

        Returns
        -------
        float
    """
    lr = cfg['learning_rate_init'] if cfg['learning_rate_init'] else 0.001
    batch_size = 200
    aug = cfg['data_aug'] if cfg['data_aug'] else "A"
    # Device configuration
    device = torch.device('cuda')
    img_width = 16
    img_height = 16

    data_transform_A = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.ToTensor()
    ])
    data_transform_B = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    data = ImageFolder(os.path.join(data_dir, "train"),
                       transform=(data_transform_A if aug == "A" else data_transform_B))

    # image size
    input_shape = (3, img_width, img_height)
    model = torchModel(cfg,
                       input_shape=input_shape,
                       num_classes=len(data.classes)).to(device)
    total_model_params = sum(p.numel() for _, p in model.state_dict().items())

    # instantiate optimizer
    model_optimizer = torch.optim.AdamW
    train_criterion = torch.nn.CrossEntropyLoss
    optimizer = model_optimizer(model.parameters(),
                                lr=lr)
    # instantiate training criterion
    train_criterion = train_criterion().to(device)

    logging.info('Generated Network:')
    summary(model, input_shape,
            device='cuda')

    num_epochs = int(np.ceil(budget))

    # Train the model
    score = []
    score_top5 = []
    score_precision = []

    # returns the cross validation accuracy
    cv = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)  # to make CV splits consistent

    for train_idx, valid_idx in cv.split(data, data.targets):
        train_data = Subset(data, train_idx)
        val_dataset = Subset(data, valid_idx)
        train_loader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True)
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=False)

        for epoch in range(num_epochs):
            logging.info('#' * 50)
            logging.info('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
            train_score, train_loss = model.train_fn(optimizer, train_criterion, train_loader, device)
            val_score, _, val_score_precision = model.eval_fn(val_loader, device)
            logging.info('Train accuracy %f', train_score)
            logging.info('Test accuracy %f', val_score)

        val_score_3, val_score_top5, val_precision = model.eval_fn(val_loader, device)

        score.append(val_score_3)
        score_top5.append(val_score_top5)
        score_precision.append(val_precision)
    # instantiate training criterion
    acc = 1 - np.mean(score)
    acc_top5 = np.mean(score_top5)
    precision = np.mean(score_precision)
    parent_directory = os.path.split(os.path.dirname(__file__))[0]
    with open(os.path.join(parent_directory, 'bohb_run0.json'), 'a')as f:
        json.dump({'configuration': dict(cfg), 'top3': np.mean(score), 'top5': acc_top5, 'precision': precision,
                   'n_params': total_model_params}, f)
        f.write("\n")

    if (precision >= constraint_precision and total_model_params <= constraint_model_size):
        # Save the model checkpoint can be restored via "model = torch.load(save_model_str)"
        torch.save(model, (os.path.join(parent_directory, str(precision))))

    return (acc, {"top5": acc_top5, "precision": precision})  # Because minimize!


if __name__ == '__main__':
    """
    This is just an example of how to implement BOHB as an optimizer!
    Here we do not consider any of the constraints and 
    """

    cmdline_parser = argparse.ArgumentParser('AutoML SS21 final project BOHB example')

    cmdline_parser.add_argument('-v', '--verbose',
                                default='INFO',
                                choices=['INFO', 'DEBUG'],
                                help='verbosity')
    cmdline_parser.add_argument('-s', "--constraint_max_model_size",
                                default=2e7,
                                help="maximal model size constraint",
                                type=int)
    cmdline_parser.add_argument('-p', "--constraint_min_precision",
                                default=0.39,
                                help='minimal constraint constraint',
                                type=float)
    cmdline_parser.add_argument('-r', "--run_id",
                                default='0',
                                help='run id ',
                                type=str)
    cmdline_parser.add_argument('-u', "--use_teset_data",
                                default='False',
                                help='check on test dataset',
                                type=bool)

    args, unknowns = cmdline_parser.parse_known_args()
    # HERE ARE THE CONSTRAINTS!
    # HERE ARE THE CONSTRAINTS!
    # HERE ARE THE CONSTRAINTS!
    constraint_model_size = args.constraint_max_model_size
    constraint_precision = args.constraint_min_precision
    use_teset_data = args.use_teset_data

    run_id = args.run_id

    logger = logging.getLogger("MLP-example")
    logging.basicConfig(level=logging.INFO)

    # Build Configuration Space which defines all parameters and their ranges.
    # To illustrate different parameter types,
    # we use continuous, integer and categorical parameters.

    cs = ConfigurationSpace()
    with open("initial_cfg.json", 'r') as f:
        opt_cfg = json.load(f)

    # We can add multiple hyperparameters at once:
    n_conf_layer = UniformIntegerHyperparameter("n_conv_layers", 1, 5, default_value=2)
    n_conf_layer_0 = UniformIntegerHyperparameter("n_channels_conv_0", 200, 1024, default_value=512)
    n_conf_layer_1 = UniformIntegerHyperparameter("n_channels_conv_1", 200, 1024, default_value=512)
    n_conf_layer_2 = UniformIntegerHyperparameter("n_channels_conv_2", 200, 1024, default_value=512)
    n_conf_layer_3 = UniformIntegerHyperparameter("n_channels_conv_3", 200, 1024, default_value=512)
    n_conf_layer_4 = UniformIntegerHyperparameter("n_channels_conv_4", 200, 1024, default_value=512)

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'micro17flower')
    learning_rate_init = UniformFloatHyperparameter('learning_rate_init',
                                                    0.000001, 1.0, default_value=2.244958736283895e-05, log=True)
    dropout = UniformFloatHyperparameter("dropout_rate", 0.1, 0.4, default_value=0.1)
    bn = CategoricalHyperparameter('use_BN', choices=[True, False])
    pooling = CategoricalHyperparameter('global_avg_pooling', choices=[True, False])

    cs.add_hyperparameters(
        [n_conf_layer, n_conf_layer_0, n_conf_layer_1, n_conf_layer_2, n_conf_layer_3, n_conf_layer_4,
         learning_rate_init, dropout, bn, pooling])

    # Add conditions to restrict the hyperparameter space
    use_conf_layer_4 = CS.conditions.InCondition(n_conf_layer_4, n_conf_layer, [5])
    use_conf_layer_3 = CS.conditions.InCondition(n_conf_layer_3, n_conf_layer, [4, 5])
    use_conf_layer_2 = CS.conditions.InCondition(n_conf_layer_2, n_conf_layer, [3, 4, 5])
    use_conf_layer_1 = CS.conditions.InCondition(n_conf_layer_1, n_conf_layer, [2, 3, 4, 5])
    # Add  multiple conditions on hyperparameters at once:
    cs.add_conditions([use_conf_layer_4, use_conf_layer_3, use_conf_layer_2, use_conf_layer_1])

    # SMAC scenario object
    scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternative to runtime)
                         "wallclock-limit": 50000,  # max duration to run the optimization (in seconds)
                         "cs": cs,  # configuration space
                         "deterministic": "False",
                         "limit_resources": True,  # Uses pynisher to limit memory and runtime
                         # Alternatively, you can also disable this.
                         # Then you should handle runtime and memory yourself in the TA
                         "memory_limit": 3072,  # adapt this to reasonable value for your hardware
                         })

    # max budget for hyperband can be anything. Here, we set it to maximum no. of epochs to train the MLP for
    max_iters = 50
    # intensifier parameters (Budget parameters for BOHB)
    intensifier_kwargs = {'initial_budget': 5, 'max_budget': max_iters, 'eta': 3}
    # To optimize, we pass the function to the SMAC-object
    smac = BOHB4HPO(scenario=scenario, rng=np.random.RandomState(42),
                    tae_runner=partial(cnn_enhanced, data_dir=data_dir, run=run_id),
                    intensifier_kwargs=intensifier_kwargs,
                    # all arguments related to intensifier can be passed like this
                    initial_design_kwargs={'n_configs_x_params': 2,  # how many initial configs to sample per parameter
                                           'max_config_fracs': .4})
    def_value = smac.get_tae_runner().run(config=opt_cfg,
                                          instance='1', budget=max_iters, seed=0)[1]
    smac.scenario.abort_on_first_run_crash = False
    # Start optimization
    try:  # try finally used to catch any interupt
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    inc_value = smac.get_tae_runner().run(config=incumbent, instance='1',
                                          budget=max_iters, seed=0)[1]

    print("Optimized Value: %.4f" % inc_value)

    list_cfg = []
    parent_directory = os.path.split(os.path.dirname(__file__))[0]
    with open(os.path.join(parent_directory, 'bohb_run0.json'), 'r')as f:
        for jsonObj in f:
            configDict = json.loads(jsonObj)
            list_cfg.append(configDict)
    precision_list = np.array([x["top3"] for x in list_cfg if
                               (x["precision"] >= constraint_precision and x['n_params'] <= constraint_model_size)])
    short_list = [x["configuration"] for x in list_cfg if
                  (x["precision"] >= constraint_precision and x['n_params'] <= constraint_model_size)]
    index = np.argmax(precision_list)
    opt_config = short_list[index]

    cs1 = ConfigurationSpace()

    n_conf_layer = UniformIntegerHyperparameter("n_conv_layers", 1, 5, default_value=4)

    n_conf_layer_0 = UniformIntegerHyperparameter("n_channels_conv_0", 200, 1024)
    n_conf_layer_1 = UniformIntegerHyperparameter("n_channels_conv_1", 200, 1024)
    n_conf_layer_2 = UniformIntegerHyperparameter("n_channels_conv_2", 200, 1024)
    n_conf_layer_3 = UniformIntegerHyperparameter("n_channels_conv_3", 200, 1024)
    n_conf_layer_4 = UniformIntegerHyperparameter("n_channels_conv_4", 200, 1024)

    list_learning = [k.get('learning_rate_init') for k in short_list]
    learning_rate_init = UniformFloatHyperparameter('learning_rate_init',
                                                    min(list_learning), (max(list_learning) * 1.1), log=True)

    drop = [k.get('dropout_rate') for k in short_list]
    dropout = UniformFloatHyperparameter("dropout_rate", 0.1, (max(drop) * 1.1))

    list_bn = [k.get('use_BN') for k in short_list]
    bn = CategoricalHyperparameter('use_BN', choices=[max(set(list_bn), key=list_bn.count)])

    aug = CategoricalHyperparameter('data_aug', choices=['A', 'B'])

    list_pooling = [k.get('global_avg_pooling') for k in short_list]
    pooling = CategoricalHyperparameter('global_avg_pooling', choices=[max(set(list_pooling), key=list_pooling.count)])

    kernel_size = CategoricalHyperparameter('kernel_size', choices=[7, 5, 3])

    cs1.add_hyperparameters(
        [n_conf_layer, n_conf_layer_0, n_conf_layer_1, n_conf_layer_2, n_conf_layer_3, n_conf_layer_4,
         learning_rate_init, dropout, bn, pooling, kernel_size, aug])

    use_conf_layer_4 = CS.conditions.InCondition(n_conf_layer_4, n_conf_layer, [5])
    use_conf_layer_3 = CS.conditions.InCondition(n_conf_layer_3, n_conf_layer, [4, 5])
    use_conf_layer_2 = CS.conditions.InCondition(n_conf_layer_2, n_conf_layer, [3, 4, 5])
    use_conf_layer_1 = CS.conditions.InCondition(n_conf_layer_1, n_conf_layer, [2, 3, 4, 5])
    # Add  multiple conditions on hyperparameters at once:
    cs1.add_conditions([use_conf_layer_4, use_conf_layer_3, use_conf_layer_2, use_conf_layer_1])

    # SMAC scenario object
    scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternative to runtime) 193
                         "wallclock-limit": 40000,  # max duration to run the optimization (in seconds)
                         "cs": cs1,  # configuration space
                         "deterministic": "False",
                         "limit_resources": True,  # Uses pynisher to limit memory and runtime
                         # Alternatively, you can also disable this.
                         # Then you should handle runtime and memory yourself in the TA
                         "memory_limit": 3072,  # adapt this to reasonable value for your hardware
                         })

    # max budget for hyperband can be anything. Here, we set it to maximum no. of epochs to train the MLP for
    max_iters = 50
    # intensifier parameters (Budget parameters for BOHB)
    intensifier_kwargs = {'initial_budget': 17, 'max_budget': max_iters, 'eta': 3}
    # To optimize, we pass the function to the SMAC-object
    smac = BOHB4HPO(scenario=scenario, rng=np.random.RandomState(42),
                    tae_runner=partial(cnn_enhanced, data_dir=data_dir, run=run_id),
                    intensifier_kwargs=intensifier_kwargs,
                    # all arguments related to intensifier can be passed like this
                    initial_design_kwargs={'n_configs_x_params': 1,  # how many initial configs to sample per parameter
                                           'max_config_fracs': .2})
    def_value = smac.get_tae_runner().run(config=opt_config,
                                          instance='1', budget=max_iters, seed=0)[1]
    smac.scenario.abort_on_first_run_crash = False
    # Start optimization
    try:  # try finally used to catch any interupt
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    inc_value = smac.get_tae_runner().run(config=incumbent, instance='1',
                                          budget=max_iters, seed=0)[1]

    print("Optimized Value: %.4f" % inc_value)

    # score your optimal configuration to disk

    final_cfg = []
    parent_directory = os.path.split(os.path.dirname(__file__))[0]
    with open(os.path.join(parent_directory, 'bohb_run0.json'), 'r')as f:
        for jsonObj in f:
            configDict = json.loads(jsonObj)
            final_cfg.append(configDict)
    precision_list = np.array([x["top3"] for x in final_cfg if
                               (x["precision"] >= constraint_precision and x['n_params'] <= constraint_model_size)])
    short_list = [x["configuration"] for x in final_cfg if
                  (x["precision"] >= constraint_precision and x['n_params'] <= constraint_model_size)]
    index = np.argmax(precision_list)
    opt_config = short_list[index]
    parent_directory = os.path.split(os.path.dirname(__file__))[0]
    with open(os.path.join(parent_directory, 'opt_cfg.json'), 'w') as f:
        json.dump(opt_config, f)

    device = torch.device('cuda')

    # check on test dataset if condition is true

    if use_teset_data == True:
        precision = precision_list[index]
        cfg = short_list[index]
        test_data = ImageFolder(os.path.join(data_dir, 'test'), transform=transforms.ToTensor())
        input_shape = (3, 16, 16)

        model = torch.load(os.path.join(parent_directory, str(precision)))
        test_loader = DataLoader(dataset=test_data,
                                 batch_size=len(test_data),
                                 shuffle=False)
        score, _, score_precision = model.eval_fn(test_loader, device)

        print("Precision on test set:", np.mean(score_precision))


