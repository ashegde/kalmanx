"""
This module contains the doppler example.
"""

from typing import Callable, Optional

import mlx.core as mx

from train import estimate_noise_covs, optimize_noise_covs
from plotting import plot_data
from src.state_space import StateSpaceModel
from src.kalman import KalmanFilter
from src.utils import sample_mvn

def create_example() -> tuple[mx.array, Callable]:
    def generate_H(x: Optional[mx.array] = None) -> mx.array:
        if x is None:
            x = mx.ones((3,))
        H = mx.zeros((4, 6))
        for i in range(3):
            H[i,i] = 1
        r = mx.linalg.norm(x)
        H[3,3:6]= x/r if r > 0 else x
        return H

    F = mx.eye(6)
    for i in range(3):
        F[i, 3+i] = 1

    return F, generate_H


def generate_trajectory(init_state: mx.array,
                        model: StateSpaceModel,
                        t_steps: int) -> tuple[mx.array, mx.array]:
        
    process_noise, observation_noise = model.generate_noise(t_steps)

    states = []
    observations = []

    state = init_state
    for t in range(t_steps):
        states.append(state)
        model.update_H(state[:3])
        state, observation = model.step(state, process_noise[t], observation_noise[t])
        observations.append(observation)
    
    return mx.stack(states), mx.stack(observations)  


def generate_dataset(init_states: mx.array,
                     model: StateSpaceModel,
                     t_steps: int) -> tuple[mx.array, mx.array]:
    x_train = []
    y_train = []

    for n in range(init_states.shape[0]):
        x, y = generate_trajectory(
            init_states[n],
            model,
            t_steps,
        )
        x_train.append(x)
        y_train.append(y)
    return mx.stack(x_train), mx.stack(y_train)


if __name__ == '__main__':

    mx.random.seed(2025)

    F, generate_H = create_example()
    H = generate_H()

    init_state_mean = mx.zeros((6,))
    init_state_cov = mx.eye(6)

    ## Simulation and training hyperparameters

    T_train = 100  # number of time steps
    N_train = 1000 # number of simulations
    batch_size = 10
    learning_rate = 1e-2
    max_iter = 500

    # test settings
    T_test = 1000
    N_test = 10

    ## Generating datasets with a "ground-truth" system

    Q_true = 0.1 * mx.eye(6)
    R_true = mx.diag(mx.array([100., 150., 60., 70]))

    truth_model = StateSpaceModel(F, H, Q_true, R_true)

    train_inits = sample_mvn(N_train, init_state_mean, init_state_cov)
    test_inits = sample_mvn(N_test, init_state_mean, init_state_cov)

    x_train, y_train = generate_dataset(train_inits, truth_model, T_train)
    x_test, y_test = generate_dataset(test_inits, truth_model, T_test)

    ## Initializing the Kalman Filters

    Q_init = mx.eye(6)
    R_init = mx.eye(4)
    
    kf = KalmanFilter(
        StateSpaceModel(F, H, Q_init, R_init),
        init_state_mean,
        init_state_cov,
    )
    Q_estimate, R_estimate = estimate_noise_covs(kf, x_train, y_train)
    kf.model.Q = Q_estimate
    kf.model.R = R_estimate


    okf = KalmanFilter(
        StateSpaceModel(F, H, Q_init, R_init),
        init_state_mean,
        init_state_cov,
    )
    Q_optimal, R_optimal = optimize_noise_covs(
        okf, x_train, y_train, learning_rate,
        batch_size, max_iter,
    )
    okf.model.Q = Q_optimal
    okf.model.R = R_optimal

    ##

    kf_mse = mx.array(0)
    okf_mse = mx.array(0)

    
    T = T_test

    for n in range(N_test):
        test_states = x_test[n]
        test_observations = y_test[n]

        okf_state_estimates = []
        kf_state_estimates = []

        okf.reset()
        kf.reset()

        for t in range(1,T):
            current_observation = test_observations[t,:] 

            kf.update_model_H(current_observation[:3])
            kf_state_mean, kf_state_cov = kf.step(current_observation)
            kf_state_estimates.append(kf_state_mean)
            kf_squared_error = mx.square(kf_state_mean - test_states[t]).sum()
            kf_mse += kf_squared_error

            okf.update_model_H(current_observation[:3])
            okf_state_mean, okf_state_cov = okf.step(current_observation)
            okf_state_estimates.append(okf_state_mean)
            okf_squared_error = mx.square(okf_state_mean - test_states[t]).sum()
            okf_mse += okf_squared_error

        okf_mse = okf_mse / (T-1)
        kf_mse = kf_mse / (T-1)

        okf_states = mx.stack(okf_state_estimates)
        kf_states = mx.stack(kf_state_estimates)

        plot_data(
            data=mx.stack([test_states[1:], kf_states, okf_states]),
            ax_titles=["x position", "y position", "z position", "x velocity", "y velocity", "z velocity"],
            labels=["Ground truth", "Estimated KF", "Optimal KF"],
            title=f"State estimates (MSE: KF = {kf_mse: 0.4f}, OKF = {okf_mse: 0.4f})",
            figsize=(18, 6),
            colors=['black', 'green', 'red'],
            linestyles=["solid", "dashed", "dashdot"],
            markers=[".", "^", "x"],
            save_path=f"states_{n}.png",
        )











