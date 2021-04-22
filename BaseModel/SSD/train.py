import argparse
import logging
import torch
import pathlib
import numpy as np
from ssd.engine.inference import do_evaluation
from ssd.config.defaults import cfg
from ssd.data.build import make_data_loader
from ssd.engine.trainer import do_train
from ssd.modeling.detector import SSDDetector
from ssd.utils.checkpoint import CheckPointer
from ssd.utils.logger import setup_logger
from ssd import torch_utils


from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)
np.random.seed(0)
torch.manual_seed(0)

# Allow torch/cudnn to optimize/analyze the input/output shape of convolutions
# To optimize forward/backward pass.
# This will increase model throughput for fixed input shape to the network
torch.backends.cudnn.benchmark = True

# Cudnn is not deterministic by default. Set this to True if you want
# to be sure to reproduce your results
torch.backends.cudnn.deterministic = True

# LR scheduler
class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.model_size = model_size
        self._rate = 0
    
    def state_dict(self):
        """Returns the state of the warmup scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
    
    def load_state_dict(self, state_dict):
        """Loads the warmup scheduler's state.
        Arguments:
            state_dict (dict): warmup scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict) 
        
    def step(self,iter_num):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate(step=iter_num)
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5))) 



def start_train(cfg):
    logger = logging.getLogger('SSD.trainer')
    model = SSDDetector(cfg)
    model = torch_utils.to_cuda(model)

    adam_optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.SOLVER.LR,
        betas=(0.9, 0.98), 
        eps=1e-09
    )

    optimizer = NoamOpt(512, 2000, adam_optimizer)
    arguments = {"iteration": 0}
    save_to_disk = True
    checkpointer = CheckPointer(
        model, optimizer, cfg.OUTPUT_DIR, save_to_disk, logger,
    )
    extra_checkpoint_data = checkpointer.load()
    arguments.update(extra_checkpoint_data)

    max_iter = cfg.SOLVER.MAX_ITER
    train_loader = make_data_loader(cfg, is_train=True, max_iter=max_iter, start_iter=arguments['iteration'])

    model = do_train(
        cfg, model, train_loader, optimizer,
        checkpointer, arguments)
    return model


def get_parser():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With PyTorch')
    parser.add_argument(
        "config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def main():
    args = get_parser().parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    output_dir = pathlib.Path(cfg.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True, parents=True)

    logger = setup_logger("SSD", output_dir)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = start_train(cfg)

    logger.info('Start evaluating...')
    torch.cuda.empty_cache()  # speed up evaluating after training finished
    do_evaluation(cfg, model)


if __name__ == '__main__':
    main()
