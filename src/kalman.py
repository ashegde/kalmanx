import mlx
import mlx.core as mx

from src.state_space import StateSpaceModel
from src.utils import linear_solve

@mx.compile
def _kalman_predict(F: mx.array,
                   state_mean: mx.array,
                   state_cov: mx.array,
                   Q: mx.array) -> tuple[mx.array, mx.array]:
        state_mean_pred = F @ state_mean
        state_cov_pred = F @ state_cov @ F.transpose() + Q
        return state_mean_pred, state_cov_pred

@mx.compile
def _kalman_update(H: mx.array,
                  state_mean: mx.array,
                  state_cov: mx.array,
                  R: mx.array,
                  observation: mx.array) -> tuple[mx.array, mx.array]:
    
    d_y = H.shape[0]
    d_x = H.shape[1]

    gain = linear_solve( 
        H @ state_cov @ H.transpose() + R,
        H @ state_cov,
    ).transpose()
    new_state_mean = state_mean + gain @ (observation - H @ state_mean) 
    new_state_cov = (mx.eye(d_x) - gain @ H) @ state_cov
    return new_state_mean, new_state_cov

@mx.compile
def kalman_step(F: mx.array,
                H: mx.array,
                Q: mx.array,
                R: mx.array,
                state_mean: mx.array,
                state_cov: mx.array,
                observation: mx.array) -> tuple[mx.array, mx.array]:
    state_mean, state_cov = _kalman_predict(
        F, state_mean, state_cov, Q,
    )
    estim_state_mean, estim_state_cov = _kalman_update(
        H, state_mean, state_cov, R, observation,
    )
    return estim_state_mean, estim_state_cov
    
class KalmanFilter:

    def __init__(self,
                 model: StateSpaceModel,
                 init_state_mean: mx.array,
                 init_state_cov: mx.array):
        
        self.model = model # internal model
        self.init_state_mean = init_state_mean
        self.init_state_cov = init_state_cov
        self.current_state_mean = init_state_mean
        self.current_state_cov = init_state_cov
        
    def step(self, observation: mx.array) -> tuple[mx.array, mx.array]:
        state_estimate_mean, state_estimate_cov = kalman_step(
             self.model.F, self.model.H, self.model.Q, self.model.R,
             self.current_state_mean, self.current_state_cov, observation,
        )
        self.current_state_mean = state_estimate_mean
        self.current_state_cov = state_estimate_cov
        return state_estimate_mean, state_estimate_cov
    
    def reset(self):
        self.current_state_mean = self.init_state_mean
        self.current_state_cov = self.init_state_cov
    
    def update_model_F(self, *args, **kwargs):
        self.model.update_F(*args, **kwargs)

    def update_model_H(self, *args, **kwargs):
        self.model.update_H(*args, **kwargs)

