import mlx
import mlx.core as mx

def sample_mvn(n_samples: int, mean: mx.array, cov: mx.array) -> mx.array:
    L = mx.linalg.cholesky(cov, stream=mx.Device(mx.cpu))
    return mx.einsum("ij,...j->...i", L, mx.random.normal((n_samples, mean.shape[0]))) + mean

@mx.compile
def cov(samples: mx.array) -> mx.array:
    """
    samples is of dimension (n_samples, d)
    """
    n = samples.shape[0]
    mcs = samples - samples.mean(axis=0, keepdims=True) # mean-centered samples
    return 1/n * mcs.T @ mcs

@mx.compile
def cholvec_to_cov(v: mx.array, d: int) -> mx.array:
    # cholvec is of dim (1/2 * d (d+1),) for some d
    L = mx.zeros((d,d))
    for i in range(d):
        for j in range(i+1):
            index = int(i*(i+1)/2 + j)
            L[i,j] = mx.exp(v[index]) if i == j else v[index]
    return L @ L.T

@mx.compile
def cov_to_cholvec(C: mx.array) -> mx.array:
    L = mx.linalg.cholesky(C, stream=mx.Device(mx.cpu))
    d = C.shape[0]
    v = mx.zeros((int(d*(d+1)/2),))
    for i in range(d):
        for j in range(i+1):
            index = int(i*(i+1)/2 + j)
            v[index] = mx.log(L[i,j]) if i == j else L[i,j]
    return v

@mx.custom_function
def linear_solve(A: mx.array, B: mx.array) -> mx.array:
    """
    Returns X s.t. AX=B 
    """
    return mx.linalg.solve(A, B, stream=mx.Device(mx.cpu))

@linear_solve.vjp
def linear_solve_vjp(primals, cotangent, output):
    A, B = primals
    grad_B = mx.linalg.solve(A.T, cotangent, stream=mx.Device(mx.cpu))
    grad_A = -grad_B @ output.T
    return grad_A, grad_B

@linear_solve.jvp
def linear_solve_jvp(primals, tangents, output):
    A, B = primals
    dA, dB = tangents
    dX = mx.linalg.solve(A, dB - dA @ output, stream=mx.Device(mx.cpu))
    return dX