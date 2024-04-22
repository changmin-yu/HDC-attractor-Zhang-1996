from typing import Optional
import numpy as np

from .utils import inverse_sigmoid, directional_tuning, sigmoid


def asymptotic_symmetric_weight(
    theta: np.ndarray, 
    a: float, 
    b: float, 
    c: float, 
    beta: float, 
    A: float, 
    B: float, 
    K: float, 
    lmd: float = 1.0, 
    lmd0: Optional[float] = None, 
    norm: str = "backward", 
):
    f = directional_tuning(theta, A, B, K)
    u = inverse_sigmoid(a, b, c, beta, f)
    
    f_fourier = np.fft.fft(f, norm=norm)
    u_fourier = np.fft.fft(u, norm=norm)
    
    if lmd0 is not None:
        lmd = lmd0 * np.max(np.square(np.abs(f_fourier)))
    else:
        lmd0 = lmd / np.max(np.square(np.abs(f_fourier)))
    
    w_fourier = f_fourier * u_fourier / (lmd + np.square(np.abs(f_fourier)))
    
    # w_fourier = np.fft.ifftshift(w_fourier)
    
    w = np.fft.ifft(w_fourier, norm=norm).real
    
    w = np.roll(w, len(w) // 2)
    
    return w, lmd0


def discrete_time_approximation(
    theta: np.ndarray, 
    f_init: np.ndarray, 
    dt: float, 
    T: float, 
    a: float, 
    b: float, 
    c: float, 
    beta: float, 
    W: Optional[np.ndarray] = None, 
    A: Optional[float] = None,
    B: Optional[float] = None, 
    K: Optional[float] = None, 
    lmd: Optional[float] = None, 
    lmd0: Optional[float] = None, 
    norm: Optional[str] = None, 
):
    t_arr = np.arange(0, T+dt, dt)
    
    if W is None:
        W, _ = asymptotic_symmetric_weight(theta, a, b, c, beta, A, B, K, lmd=lmd, lmd0=lmd0, norm=norm)
    
    u_arr = np.zeros((len(t_arr), len(f_init)))
    f_arr = np.zeros((len(t_arr), len(f_init)))
    f_arr[0] = f_init
    u_arr[0] = inverse_sigmoid(a, b, c, beta, f_init)
    
    for i in range(1, len(t_arr)):
        # u_arr[i] = np.convolve(W, f_arr[i-1], mode="same")
        u_arr[i] = np.real(np.fft.ifft(np.fft.fft(W, norm=norm)*np.fft.fft(f_arr[i-1], norm=norm), norm=norm))
        f_arr[i] = sigmoid(a, b, c, beta, u_arr[i])
    
    return u_arr, f_arr


if __name__=="__main__":
    theta = np.linspace(-np.pi, np.pi, num=200, endpoint=True)
    
    a = 6.34
    b = 10
    c = 0.5
    beta = 0.8
    K = 8
    A = 1
    f_max = 40 
    B = (f_max - A) / np.exp(K)
    
    f_init = np.random.uniform(0, 1, size=(len(theta), ))
    
    W, lmd0 = asymptotic_symmetric_weight(theta, a, b, c, beta, A, B, K, lmd0=1e-3, norm="forward")

    u_arr = discrete_time_approximation(
        theta=theta, 
        f_init=f_init, 
        dt=10, 
        T=500, 
        a=a, 
        b=b, 
        c=c, 
        beta=beta, 
        W=W, 
    )