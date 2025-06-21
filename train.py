"""
This module contains functionality for estimating and optimizing the Kalman filter
parameters -- i.e., the noise covariance matrices. 
"""

import mlx
import mlx.core as mx
import mlx.optimizers as optim
from tqdm import tqdm 

from src.kalman import KalmanFilter, kalman_step
from src.utils import cov, cholvec_to_cov, cov_to_cholvec

def estimate_noise_covs(kalman_filter: KalmanFilter,
                        x_train: mx.array,
                        y_train: mx.array) -> tuple[mx.array, mx.array]: 
    N, T, _ = x_train.shape
    state_error = []
    observation_error = []
    # Test estimate
    for n in range(N):
        for t in range(T-1):
            current_state = x_train[n,t,:]  # (d_x,)
            next_state = x_train[n,t+1,:]  # (d_x,)
            current_observation = y_train[n,t,:]  # (d_y,)
            kalman_filter.update_model_H(current_observation[:3])

            state_error.append(next_state - kalman_filter.model.F @ current_state)
            observation_error.append(current_observation - kalman_filter.model.H @ current_state)

    Q_estimate = cov(mx.stack(state_error).reshape((-1, kalman_filter.model.d_x)))
    R_estimate = cov(mx.stack(observation_error).reshape((-1, kalman_filter.model.d_y)))

    return Q_estimate, R_estimate

def optimize_noise_covs(kalman_filter: KalmanFilter,
                        x_train: mx.array,
                        y_train: mx.array,
                        learning_rate: float,
                        batch_size: int,
                        max_iter: int) -> tuple[mx.array, mx.array]:
    
    def loss_fn(params: dict, x_train: mx.array, y_train: mx.array):
        loss = mx.array(0)
        Q = cholvec_to_cov(params["w_q"], kalman_filter.model.d_x)
        R = cholvec_to_cov(params["w_r"], kalman_filter.model.d_y)
        M, T, _ = x_train.shape

        for m in range(M):
            state_mean = kalman_filter.init_state_mean
            state_cov = kalman_filter.init_state_cov
            for t in range(1, T):
                current_measurement = y_train[m,t,:]
                kalman_filter.update_model_H(current_measurement[:3])
                state_mean, state_cov = kalman_step(
                    kalman_filter.model.F, kalman_filter.model.H, Q, R,
                    state_mean, state_cov, current_measurement,
                )
                loss += mx.square(state_mean - x_train[m, t]).sum()
        return loss / (M*(T-1))


    Q_estimate, R_estimate = estimate_noise_covs(kalman_filter, x_train, y_train)
    params = {
        "w_q": cov_to_cholvec(Q_estimate),
        "w_r": cov_to_cholvec(R_estimate),
    }

    lr_scheduler = optim.step_decay(learning_rate, decay_rate=0.99, step_size=50)
    optimizer = optim.Adam(lr_scheduler)

    with open(f"optim.log", "w") as file:
        file.write("Losses for optimal KF\n")

    N = x_train.shape[0]
    for _ in (pbar:=tqdm(range(max_iter))):
        indices = mx.random.permutation(N)[:batch_size]
        loss, grads = mx.value_and_grad(loss_fn)(params, x_train[indices], y_train[indices])
        optimizer.update(params, grads)
        mx.eval(params, optimizer.state)
        pbar.set_postfix_str(f"loss: {loss: 0.5e} | lr: {optimizer.learning_rate: 0.5e}")
        
        # logging
        with open(f"optim.log", "a") as file:
                file.write(f"{loss}\n")

    Q_optimal = cholvec_to_cov(params["w_q"], kalman_filter.model.d_x)
    R_optimal = cholvec_to_cov(params["w_r"], kalman_filter.model.d_y)

    return Q_optimal, R_optimal