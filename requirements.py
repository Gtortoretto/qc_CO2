# utils.py

# General imports
import warnings
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from IPython import get_ipython
from tqdm import tqdm
from scipy.interpolate import make_interp_spline


# Qiskit imports
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector

# Qiskit Nature
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_algorithms import VQE
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import SLSQP

# Qiskit Aer
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Estimator as AerEstimator




def configure_plotting():

    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor':   '#282c34',
        'axes.facecolor':     '#1e1e1e',
        'grid.color':         '#444444',
        'figure.figsize':     (10, 6),
        'figure.dpi':         150
    })

def suppress_warnings():

    warnings.filterwarnings('ignore', category=DeprecationWarning)

configure_plotting()
suppress_warnings()


def get_current_directory():

    try:
        return os.path.dirname(os.path.abspath(__file__))
    
    except NameError:
        return os.getcwd()

def _ensure_data_dir():
    
    base_dir = get_current_directory()
    data_dir = os.path.join(base_dir, 'data') 
    os.makedirs(data_dir, exist_ok=True)
    return data_dir

def save(obj_name_str):

    caller_globals = get_ipython().user_ns if get_ipython() else globals()
    
    if obj_name_str not in caller_globals:
        raise ValueError(f"Variable '{obj_name_str}' not found in current scope.")
    
    obj = caller_globals[obj_name_str]
    save_obj(obj, obj_name_str)

def load(filename):

    data_dir = _ensure_data_dir()
    file_path = os.path.join(data_dir, f'{filename}.pickle')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_obj(obj, filename):

    data_dir = _ensure_data_dir()
    file_path = os.path.join(data_dir, f'{filename}.pickle')
    
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
