from joblib import load
import numpy as np
from utils.parser import parse_args
from pymoo.optimize import minimize
from scripts.models3 import MyProblem, MyCallback


args = parse_args()
checkpoint, = np.load(args.checkpoint_path, allow_pickle=True).flatten()

print(-checkpoint.result().F)
print(checkpoint.result().X)
print(checkpoint)
