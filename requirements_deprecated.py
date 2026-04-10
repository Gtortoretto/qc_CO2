# utils.py

# General imports

import warnings
import os
import pickle
import time
import numpy as np
import seaborn as sns
import pandas as pd
import inspect
from pathlib import Path
from typing import Any, Optional
from IPython import get_ipython
from tqdm import tqdm

from scipy.interpolate import make_interp_spline, UnivariateSpline
from scipy.optimize import curve_fit, minimize

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

from pyscf import gto, scf, dft, fci

# Qiskit imports
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector

# Qiskit Nature
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock

# Qiskit Primitives & Algorithms
from qiskit.primitives import BaseEstimatorV1       
from qiskit.primitives import Estimator as EstimatorV1                  
from qiskit.primitives import BackendEstimator as BackendEstimatorV1
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import SLSQP, SPSA, COBYLA, NELDER_MEAD, AQGD, L_BFGS_B, P_BFGS, TNC
from qiskit_algorithms import VQE, NumPyMinimumEigensolver

# Qiskit Aer
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

# Qiskit Runtime
from qiskit_ibm_runtime.fake_provider import FakeManilaV2, FakeJakartaV2, FakeTorino, FakeKyoto, FakeOsaka, FakeBrisbane


# --- Style settings ---

def set_style(style="production"):

    # Style : dark 

    if style == "dark":
        
        plt.style.use('dark_background')
        
        plt.rcParams.update({
            'font.size': 14,
            'axes.labelsize': 16,
            'axes.titlesize': 18,
            'lines.linewidth': 2.5,
            'axes.grid': True,
            'grid.alpha': 0.2,
            'grid.linestyle': '--',
            'figure.figsize': (10, 6),
            'legend.frameon': False,  
            'figure.dpi': 150
        })

        print("Style: Dark")
        
    # Style : production 
    
    elif style == "production":

        plt.style.use('default') 
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': 14,
            'axes.labelsize': 16,
            'axes.titlesize': 18,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 13,
            'figure.figsize': (8, 6),
            'lines.linewidth': 2.5,
            'lines.markersize': 8,
            'axes.linewidth': 1.5,
            'xtick.major.width': 1.5,
            'ytick.major.width': 1.5,
            'xtick.direction': 'in', 
            'ytick.direction': 'in',
            'xtick.top': True,       
            'ytick.right': True,
            'figure.dpi': 300,       
            'savefig.dpi': 300,
            'savefig.bbox': 'tight', 
            'savefig.pad_inches': 0.1
        })
        print("Style: Production")


def suppress_warnings():

    warnings.filterwarnings('ignore', category=DeprecationWarning)

set_style("production")
suppress_warnings()

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

IMG_PNG_DIR = os.path.join(PROJECT_ROOT, "1. Data", "Images", "PNG")
IMG_PDF_DIR = os.path.join(PROJECT_ROOT, "1. Data", "Images", "PDF")
PICKLE_DIR = os.path.join(PROJECT_ROOT, "1. Data", "Pickles")

for d in [PICKLE_DIR, IMG_PNG_DIR, IMG_PDF_DIR]:
    os.makedirs(d, exist_ok=True)

def save_data(data, filename):

    if not filename.endswith('.pkl'):
        filename += '.pkl'
        
    filepath = os.path.join(PICKLE_DIR, filename)
    
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Saved to: {os.path.relpath(filepath, PROJECT_ROOT)}")
    except Exception as e:
        print(f"✘ Error saving {filename}: {e}")

def load_data(filename):

    if not filename.endswith('.pkl'):
        filename += '.pkl'
        
    filepath = os.path.join(PICKLE_DIR, filename)
    
    if not os.path.exists(filepath):
        print(f"✘ File not found: {filepath}")
        return None
        
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ Loaded: {filename}")
        return data
    except Exception as e:
        print(f"✘ Error loading {filename}: {e}")
        return None
    

def save_png(filename, fig=None):
    
    if not filename.endswith('.png'): filename += '.png'
    filepath = os.path.join(IMG_PNG_DIR, filename)

    target_fig = fig if fig is not None else plt.gcf()
    
    try:
        target_fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved PNG: {filename}")
    except Exception as e:
        print(f"✘ Error saving: {e}")

def save_pdf(filename, fig=None):

    if not filename.endswith('.pdf'): filename += '.pdf'
    filepath = os.path.join(IMG_PDF_DIR, filename)
    
    target_fig = fig if fig is not None else plt.gcf()
    
    try:
        target_fig.savefig(filepath, bbox_inches='tight')
        print(f"✓ Saved PDF: {filename}")
    except Exception as e:
        print(f"✘ Error saving: {e}")