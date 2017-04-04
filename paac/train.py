import argparse
import logging
import sys

import environment_creator
from paac import PAACLearner
from policy_v_network import NaturePolicyVNetwork, NIPSPolicyVNetwork, SurpriseExplorationNetwork
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def bool_arg(string):
    value = string.lower()
    if value == 'true':
        return True
    elif value == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError("Expected True or False, but got {}".format(string))


def main(args):
    logging.debug('Configuration: {}'.format(args))

    network, env_creator = get_network_and_environment_creator(args)
    learner = PAACLearner(network, env_creator, args)

    logging.info('Starting training')
    learner.train()
    logging.info('Finished training')


def get_network_and_environment_creator(args, random_seed=3):
    env_creator = environment_creator.EnvironmentCreator(args)
    num_actions = env_creator.num_actions
    args.num_actions = num_actions
    args.random_seed = random_seed

    network_conf = {'name': "local_learning",
                    'num_actions': num_actions,
                    'entropy_regularisation_strength': args.entropy_regularisation_strength,
                    'device': args.device,
                    'emulator_counts': args.emulator_counts,
                    'clip_norm': args.clip_norm,
                    'clip_norm_type': args.clip_norm_type}
    if args.arch == 'NIPS':
        network = NIPSPolicyVNetwork(network_conf)
    elif args.arch == 'SURP':
        network = SurpriseExplorationNetwork(network_conf)
    else:
        network = NaturePolicyVNetwork(network_conf)
    return network, env_creator


def get_arg_parser():
    parser = argparse.ArgumentParser()
    #pong
    #breakout
    #qbert
    #freeway
    #frostbite
    #montezuma_revenge
    parser.add_argument('-g', default='breakout', help='Name of game', dest='game')
    parser.add_argument('-d', '--device', default='/gpu:0', type=str, help="Device to be used ('/cpu:0', /gpu:0, /gpu:1,...)", dest="device")
    parser.add_argument('--rom_path', default='/home/mikkel/ALE_roms/', help='Directory where the game roms are located (needed for ALE environment)', dest="rom_path")
    parser.add_argument('-v', '--visualize', default=False, type=bool_arg, help="0: no visualization of emulator; 1: all emulators, for all actors, are visualized; 2: only 1 emulator (for one of the actors) is visualized", dest="visualize")
    parser.add_argument('--e', default=0.1, type=float, help="Epsilon for the Rmsprop and Adam optimizers", dest="e")
    parser.add_argument('--alpha', default=0.99, type=float, help="Discount factor for the history/coming gradient, for the Rmsprop optimizer", dest="alpha")
    parser.add_argument('-lr', '--initial_lr', default=0.0224, type=float, help="Initial value for the learning rate. Default = LogUniform(10**-4, 10**-2)", dest="initial_lr")
    parser.add_argument('-lra', '--lr_annealing_steps', default=80000000, type=int, help="Nr. of global steps during which the learning rate will be linearly annealed towards zero", dest="lr_annealing_steps")
    parser.add_argument('--entropy', default=0.01, type=float, help="Strength of the entropy regularization term (needed for actor-critic)", dest="entropy_regularisation_strength")
    parser.add_argument('--clip_norm', default=40, type=float, help="If clip_norm_type is local/global, grads will be clipped at the specified maximum (avaerage) L2-norm", dest="clip_norm")
    parser.add_argument('--clip_norm_type', default="global", help="Whether to clip grads by their norm or not. Values: ignore (no clipping), local (layer-wise norm), global (global norm)", dest="clip_norm_type")
    parser.add_argument('--gamma', default=0.99, type=float, help="Discount factor", dest="gamma")
    parser.add_argument('--max_global_steps', default=20000000, type=int, help="Max. number of training steps", dest="max_global_steps")
    parser.add_argument('--max_local_steps', default=5, type=int, help="Number of steps to gain experience from before every update for the Q learning/A3C algorithm", dest="max_local_steps")
    parser.add_argument('--arch', default='SURP', help="Which network architecture to use: from the NIPS, NATURE paper or SURP (for surprise-based exploration)", dest="arch")
    parser.add_argument('--single_life_episodes', default=False, type=bool_arg, help="If True, training episodes will be terminated when a life is lost (for games)", dest="single_life_episodes")
    parser.add_argument('-ec', '--emulator_counts', default=32, type=int, help="The amount of emulators per agent. Default is 1.", dest="emulator_counts")
    parser.add_argument('-ew', '--emulator_workers', default=8, type=int, help="The amount of emulator workers per agent. Default is 1.", dest="emulator_workers")
    parser.add_argument('-df', '--debugging_folder', default='debugging', type=str, help="Folder where to save the debugging information.", dest="debugging_folder")
    parser.add_argument('-rs', '--random_start', default=True, type=bool_arg, help="Whether or not to start with 30 noops for each env. Default True", dest="random_start")
    parser.add_argument('-af', '--arg_file', default=None, type=str, help="Path to the file from which to load args", dest="arg_file")
    return parser


if __name__ == '__main__':
    args = get_arg_parser().parse_args()

    import custom_logging

    # Override command line args with args from file if supplied
    if args.arg_file is not None:
        for k, v in custom_logging.load_args(args.arg_file).items():
            setattr(args, k, v)

    custom_logging.save_args(args, args.debugging_folder)
    logging.debug(args)

    main(args)
