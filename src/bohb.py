"""
===========================
Optimization using BOHB
===========================
"""

import logging
import os
import json
import ConfigSpace as CS
import numpy as np
import argparse
from functools import partial

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant
from sklearn.model_selection import StratifiedKFold

from smac.configspace import ConfigurationSpace
from smac.facade.smac_bohb_facade import BOHB4HPO
from smac.scenario.scenario import Scenario

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchsummary import summary
from torchvision.datasets import ImageFolder

from cnn import torchModel


def get_optimizer_and_crit(cfg):
    if cfg['optimizer'] == 'AdamW':
        model_optimizer = torch.optim.AdamW
    else:
        model_optimizer = torch.optim.Adam

    if cfg['train_criterion'] == 'mse':
        train_criterion = torch.nn.MSELoss
    else:
        train_criterion = torch.nn.CrossEntropyLoss
    return model_optimizer, train_criterion


# Target Algorithm
# The signature of the function determines what arguments are passed to it
# i.e., budget is passed to the target algorithm if it is present in the signature
def cnn_from_cfg(cfg, seed, instance, budget ,run='1',
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
    batch_size = cfg['batch_size'] if cfg['batch_size'] else 200
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img_width = 16
    img_height = 16
    data_augmentations = transforms.ToTensor()

    data = ImageFolder(os.path.join(data_dir, "train"), transform=data_augmentations)
    test_data = ImageFolder(os.path.join(data_dir, "test"), transform=data_augmentations)
    targets = data.targets

    # image size
    input_shape = (3, img_width, img_height)

    model = torchModel(cfg,
                       input_shape=input_shape,
                       num_classes=len(data.classes)).to(device)
    total_model_params = np.sum(p.numel() for p in model.parameters())
    # if total_model_params < 1e5 or total_model_params == 1e5:
    # instantiate optimizer
    model_optimizer, train_criterion = get_optimizer_and_crit(cfg)
    optimizer = model_optimizer(model.parameters(),
                                    lr=lr)
    # instantiate training criterion
    train_criterion = train_criterion().to(device)

    logging.info('Generated Network:')
    summary(model, input_shape,
                device='cuda' if torch.cuda.is_available() else 'cpu')

    num_epochs = int(np.ceil(budget))

    # Train the model
    score = []
    score_top5 = []
    score_precision = []

    # returns the cross validation accuracy
    cv = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)  # to make CV splits consistent
    # for train_idx, valid_idx in cv.split(data, data.targets):
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
                val_score, val_score_top5, val_score_precision = model.eval_fn(val_loader, device)
                logging.info('Train accuracy %f', train_score)
                logging.info('Test accuracy %f', val_score)
        score.append(val_score)
        score_top5.append(val_score_top5)
        score_precision.append(val_score_precision)
    # instantiate training criterion
    acc = 1 - np.mean(score)
    acc_top5 = np.mean(score_top5)
    precision = np.mean(score_precision)
    with open('bohb_run%s.json'% run, 'a+')as f:
        json.dump({'configuration': dict(cfg), 'top3': np.mean(score), 'top5': acc_top5, 'precision': precision, 'n_params': total_model_params}, f)
        f.write("\n")
    # else:
    #     acc = 1

    return (acc, {"top5":acc_top5, "precision": precision})  # Because minimize!


if __name__ == '__main__':
    """
    This is just an example of how to implement BOHB as an optimizer!
    Here we do not consider any of the constraints and 
    """

    cmdline_parser = argparse.ArgumentParser('AutoML SS21 final project BOHB example')

    cmdline_parser.add_argument('-e', '--epochs',
                                default=50,
                                help='Number of epochs',
                                type=int)
    cmdline_parser.add_argument('-m', '--model_path',
                                default=None,
                                help='Path to store model',
                                type=str)
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

    args, unknowns = cmdline_parser.parse_known_args()
    # HERE ARE THE CONSTRAINTS!
    # HERE ARE THE CONSTRAINTS!
    # HERE ARE THE CONSTRAINTS!
    constraint_model_size = args.constraint_max_model_size
    constraint_precision = args.constraint_min_precision

    run_id = args.run_id

    logger = logging.getLogger("MLP-example")
    logging.basicConfig(level=logging.INFO)

    # Build Configuration Space which defines all parameters and their ranges.
    # To illustrate different parameter types,
    # we use continuous, integer and categorical parameters.
    cs = ConfigurationSpace()

    # We can add multiple hyperparameters at once:
    n_conf_layer = UniformIntegerHyperparameter("n_conv_layers", 1, 3, default_value=3)
    n_conf_layer_0 = UniformIntegerHyperparameter("n_channels_conv_0", 512, 2048, default_value=512)
    n_conf_layer_1 = UniformIntegerHyperparameter("n_channels_conv_1", 512, 2048, default_value=512)
    n_conf_layer_2 = UniformIntegerHyperparameter("n_channels_conv_2", 512, 2048, default_value=512)

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'micro17flower')
    learning_rate_init = UniformFloatHyperparameter('learning_rate_init',
                                                    0.00001, 1.0, default_value=2.244958736283895e-05, log=True)
    cs.add_hyperparameters([n_conf_layer, n_conf_layer_0, n_conf_layer_1, n_conf_layer_2,
                            learning_rate_init])

    # Add conditions to restrict the hyperparameter space
    use_conf_layer_2 = CS.conditions.InCondition(n_conf_layer_2, n_conf_layer, [3])
    use_conf_layer_1 = CS.conditions.InCondition(n_conf_layer_1, n_conf_layer, [2, 3])
    # Add  multiple conditions on hyperparameters at once:
    cs.add_conditions([use_conf_layer_2, use_conf_layer_1])

    # SMAC scenario object
    scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternative to runtime)
                         "wallclock-limit": 43200,  # max duration to run the optimization (in seconds)
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
                    tae_runner=partial(cnn_from_cfg, data_dir=data_dir, run=run_id),
                    intensifier_kwargs=intensifier_kwargs,  # all arguments related to intensifier can be passed like this
                    initial_design_kwargs={'n_configs_x_params': 1,  # how many initial configs to sample per parameter
                                           'max_config_fracs': .2})
    # def_value = smac.get_tae_runner().run(config=cs.get_default_configuration(),
    #                                       instance='1', budget=max_iters, seed=0)[1]
    # Start optimization
    try:  # try finally used to catch any interupt
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    inc_value = smac.get_tae_runner().run(config=incumbent, instance='1',
                                          budget=max_iters, seed=0)[1]
    print("Optimized Value: %.4f" % inc_value)

    # srore your optimal configuration to disk
    opt_config = incumbent.get_dictionary()
    with open('opt_cfg.json', 'w') as f:
        json.dump(opt_config, f)
