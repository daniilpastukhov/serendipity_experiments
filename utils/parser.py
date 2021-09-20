import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='AutoML RS')
    parser.add_argument(
        '--dataset_path',
        help='Path to dataset folder',
        type=str
    )
    parser.add_argument(
        '--checkpoint_path',
        help='Path to checkpoint',
        default=None,
        type=str
    )
    parser.add_argument(
        '--save_path',
        help='Checkpoints save folder',
        type=str
    )
    parser.add_argument(
        '--algorithm',
        help='Algorithm (random_search, ga, nsga2)',
        type=str,
        default='nsga2'
    )
    parser.add_argument(
        '--n_obj', metavar='N',
        help='Number of objectives',
        type=int,
        default=1
    )
    parser.add_argument(
        '--i',
        help='Current iteration, only if checkpoint was loaded',
        type=int,
        default=1
    )
    parser.add_argument(
        '--pop_size',
        help='Population size',
        type=int,
        default=7
    )
    parser.add_argument(
        '--n_gen',
        help='Number of generations',
        type=int,
        default=5
    )
    parser.add_argument(
        '--res_path',
        help='Path to res.joblib',
        type=str,
        default=None
    )
    parser.add_argument(
        '--alpha',
        help='Novelty weight',
        type=float,
        default=0.3
    )
    parser.add_argument(
        '--beta',
        help='Unexpectedness weight',
        type=float,
        default=0.4
    )
    parser.add_argument(
        '--gamma',
        help='Relevance weight',
        type=float,
        default=0.3
    )

    args = parser.parse_args()
    return args
