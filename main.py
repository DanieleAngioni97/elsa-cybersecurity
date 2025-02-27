"""
nohup python -u main.py --track 1 > nohups/nohup.out &
nohup python -u main.py --track 2 > nohups/nohup.out &
"""
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'android-detectors/src/'))

import argparse
import json
import config
from loaders import svmcb_loader
import itertools
import multiprocessing
from datetime import datetime

def myprint(text):
    now = datetime.now().strftime("[%Y-%m-%d/%H:%M:%S]")
    print(f"[{now}] {text}", flush=True)

def run(process_args):
    method_name, b, n_unst, max_it, track = process_args

    method_hparam_name = f"{method_name}-b{b}-n{n_unst}-it{max_it}"
    myprint(f"Starting {method_hparam_name}")

    classifier = svmcb_loader.load(clf_name=method_hparam_name,
                                   b=b,
                                   n_unst=n_unst,
                                   max_it=max_it)  # load_classifier(args)
    myprint(f"Finished training {method_hparam_name}")

    myprint(f"Evaluating track {track} for {method_hparam_name}")
    track_module = f"track_{track}.evaluation"
    evaluate = __import__(track_module, fromlist=["evaluate"]).evaluate
    results = evaluate(classifier, config)
    with open(os.path.join(
            f"submissions/submission_{method_hparam_name}_"
            f"track_{track}.json"), "w") as f:
        json.dump(results, f)

    myprint(f"Finished track {track} for {method_hparam_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clf_loader_path",
                        type=str,
                        default='android-detectors/src/loaders/svmcb_loader.py',
                        help="Path of a Python module containing a `load()` method that "
                             "returns a trained classifier.")
    parser.add_argument("--track",
                        type=int, choices=[1, 2, 3],
                        default=1,
                        help="Evaluation track for which to produce the "
                             "submission file.")
    parser.add_argument("--method_name",
                        type=str,
                        default='SVM-CB',
                        help="Name of the detection algorithm.")
    # Hyperparameters of SVM-CB
    parser.add_argument("--b",
                        type=float, default=[0.2, 0.8], #default=[0.2, 0.8],
                        nargs='+',
                        help="Weight (abs) bounding value.")
    parser.add_argument("--n_unst",
                        type=int, default=[100], nargs='+',
                        help="Number of unstable features to be bounded.")
    parser.add_argument("--max_it",
                        type=float,
                        default=10000,
                        help="Maximum number of iterations.")
    args = parser.parse_args()

    b_list, n_unst_list = args.b, args.n_unst
    hparams = itertools.product(b_list, n_unst_list)
    # Prepare arguments for multiprocessing
    process_args = [(args.method_name, b, n_unst, args.max_it, args.track) for b, n_unst in hparams]

    print(f"Track {args.track}")

    #Use multiprocessing to parallelize execution with different hyperparameters
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(run, process_args)


if __name__ == "__main__":
    main()
