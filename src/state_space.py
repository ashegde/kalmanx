from typing import Callable, Optional

import mlx.core as mx
from src.utils import sample_mvn

class NoUpdateFunctionException(Exception): pass

@mx.compile
def _forward_step(state: mx.array,
                  F: mx.array,
                  H: mx.array,
                  Q: mx.array,
                  R: mx.array) -> tuple[mx.array, mx.array]:
    # state is (d_x, )
    # Q is (d_x, )
    # R is (d_y, )
    next_state = F @ state + Q    # (B, d_x)
    observation = H @ state + R  # (B, d_y)
    return next_state, observation


class StateSpaceModel:
    """
    State Space Model

    x_{t+1} = F_t x_t + w_t
    y_t = H_t x_t + v_t
    where w_t ~_{iid} N(0, Q) and v_t ~_{iid} N(0, R),
    and x0 ~ N(m, P)

    Varying F_t and H_t are handled manually by updating.
    """

    def __init__(self,
                 F: mx.array,
                 H: mx.array,
                 Q: mx.array,
                 R: mx.array,
                 update_F_fn: Optional[Callable] = None,
                 update_H_fn: Optional[Callable] = None):
        
        # F, H are initial transition and observation matrices
        #  
        self.F, self.H, self.Q, self.R = F, H, Q, R
        self.update_F_fn = update_F_fn
        self.update_H_fn = update_H_fn
        self.d_x = F.shape[1]
        self.d_y = H.shape[0]
    

    def update_F(self, *args, **kwargs):
        if self.update_F_fn:
            self.F = self.update_F_fn(*args, **kwargs)
        else:
            NoUpdateFunctionException("No function to update F specified.")

    def update_H(self, *args, **kwargs):
        if self.update_H_fn:
            self.H = self.update_H_fn(*args, **kwargs)
        else:
            NoUpdateFunctionException("No function to update H specified.")
    
    def generate_noise(self, num_samples: int) -> tuple[mx.array, mx.array]:

        process_noise = sample_mvn(num_samples, mx.zeros(self.d_x,), self.Q)
        observation_noise = sample_mvn(num_samples, mx.zeros(self.d_y,), self.R)

        return process_noise, observation_noise

    def step(self,
             state: mx.array,
             process_noise: mx.array,
             observation_noise: mx.array) -> tuple[mx.array, mx.array]:
        next_state, observation = _forward_step(
            state,
            self.F,
            self.H,
            process_noise,
            observation_noise,
        )  # next_state is (B, d_x), observation is (B, d_y)

        return next_state, observation 
  