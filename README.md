## Kalmanx

This folder provides some scripts for implementing Kalman filters (KFs) and optimal KFs in MLX. The main reference here is:

```
Greenberg, I., Yannay, N., & Mannor, S. (2023). Optimization or architecture: How to hack kalman filtering. Advances in Neural Information Processing Systems, 36, 50482-50505.
```

The prepared example can be run simply by running ```python doppler.py``` in the terminal. Doing so produces the following result:

![states_0](https://github.com/user-attachments/assets/8dbfaac3-064d-4780-bf2e-767f098be733)

These scripts are still work-in-progress and so there are several caveats. First, the code is not as efficient as it could be -- all of the linear solves are performed on the CPU.
For simplicity, I have chosen not to implement batch processing into the state space models. This makes handling time-varying state-transition and observation matrices very simple.
But this can be easily rectified. Second, the loss appears to be somewhat difficult to optimize or perhaps just requires a larger number of iterations. In the ```notebooks``` folder, 
I have included a notebook containing functionality for verifying the loss function gradients by comparing with finite differences.
It appears that at small epsilon (<1e-3), non-negligible differences can manifest due to the numerics of the linear solve (and backprop through it).
Due to these differences, the optimal KF as implemented may not always outperform the classical strategy of empirically estimating the Q and R covariance matrices.
Oddly enough, in the example above, the optimal KF strategy picks Q and R in such a way that the estimated states don't blow up.
