"""
Copyright © 2024 Chun Hei Michael Chan, MIPLab EPFL
"""

from src.utils import *
from src.metrics import *
from src.operations import *


def spectral_filter_directed(signal:np.ndarray, kernel:np.ndarray, 
                    U:np.ndarray, V:np.ndarray, Uinv:Optional[np.ndarray]=None):
    """
    Applies a graph filter to a signal on a directed graph.
    
    Paramters
    ---------
        signal (np.ndarray): The input signal to be filtered.
        kernel (np.ndarray): The graph filter kernel.
        U (np.ndarray): The eigenvectors of the graph Laplacian.
        V (np.ndarray): The eigenvectors of the graph Laplacian.
        Uinv (Optional[np.ndarray]): The inverse of the eigenvectors of the graph Laplacian.
    
    Returns
    ---------
        np.ndarray: The filtered signal.
    """
        
    filtered_sig = inverseGFT(kernel @ GFT(signal, U, Uinv=Uinv), U)
    return filtered_sig

def vandermonde_matrix(v:np.ndarray, dim:int):
    """
    Computes the Vandermonde matrix of a vector.

    Paramters
    ---------
        v (np.ndarray): The vector to compute the Vandermonde matrix of.
        dim (int): The dimension of the Vandermonde matrix.
    Returns
    ---------
    np.ndarray: The Vandermonde matrix.
    """
    vdm = np.zeros((len(v), dim)).astype(complex)
    for sidx in range(dim):
        vdm [:, sidx] = v ** sidx
    return vdm

def get_polynomial_coefficients(A:np.ndarray, kernel:np.ndarray,
                                V:np.ndarray, minpolydeg:float, 
                                normalize_gso:bool=True):
    """
    
    Simply solve for (c_i) the system spectral with filter P (i.e kernel) and A=UVU^{-1}
    P = \sum_i\geq 0 c_i V^i
    Paramters
    ---------
    
    Returns
    ---------

    """
    if normalize_gso:
        Vnorm = V / np.abs(V).max()
    else:
        Vnorm = V / 1.0
    vdm = vandermonde_matrix(Vnorm, minpolydeg)
    # c = np.linalg.inv(vdm) @ kernel
    c = np.linalg.pinv(vdm) @ kernel

    return vdm, c