import argparse
from my_functions import *
import pprint

import warnings
warnings.filterwarnings('ignore')
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

parser = argparse.ArgumentParser(description='Flower Classifier')

parser.add_argument("image_path")
parser.add_argument("model_file")
parser.add_argument('--top_k', dest="top_k", type=int, default=5)
parser.add_argument('--category_names', dest="label_names", default='./label_map.json')

args = parser.parse_args()

probs, classes = predict(args.image_path, args.model_file, args.label_names, args.top_k)

pprint.pprint(list(zip(classes, probs)))
quit()