import os
import json
import joblib
import tensorflow as tf
import argparse
from maml_zoo.samplers.utils import rollout


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_path_length', type=int, default=1000,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    parser.add_argument('--video_filename', type=str,
                        help='path to the out video file')
    parser.add_argument('--prompt', type=bool, default=False,
                        help='Whether or not to prompt for more sim')
    parser.add_argument('--ignore_done', type=bool, default=False,
                        help='Whether stop animation when environment done or continue anyway')
    args = parser.parse_args()

    # If the snapshot file use tensorflow, do:
    # import tensorflow as tf
    # with tf.Session():
    #     [rest of the code]
    paths = []
    jsons = []
    d = '../pearl-baselines'
    for r, d, files in os.walk(d):
        for f in files:
            if 'pkl' in f:
                paths.append(os.path.join(r, f))
            elif 'json' in f:
                jsons.append(os.path.join(r, f))

    for p, j in zip(paths, jsons):
        with tf.Session() as sess:
            js = json.load(open(j))
            if 'Cheetah' in js['env']['$class']:
                pkl_path = p
                video_filename = p.split('/')[-1] + '.mp4'
                print("Testing policy %s" % pkl_path)
                data = joblib.load(pkl_path)
                policy = data['policy']
                policy._pre_update_mode = True
                env = data['env']
                path = rollout(env, policy, max_path_length=args.max_path_length, animated=True, speedup=args.speedup,
                            video_filename=video_filename, save_video=True, ignore_done=args.ignore_done)
