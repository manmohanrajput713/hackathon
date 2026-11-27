# Gas-lift skip PINC

Physics-informed neural net for control of a gas-lifted oil well. The model is a feedforward network. Time is an input. Derivatives in the physics loss are automatic derivatives with respect to that time input. The plant is the explicit Jahanshahi ODE in Kittelsen, Antonelo, Camponogara, and Imsland (2024). There is no official reference implementation. This repository does not use an LSTM, a batch-axis finite-difference `dydt`, or the ESP `ed_bcs` residual.

```
@article{kittelsen2024gaslift,
  title   = {Physics-Informed Neural Networks with Skip Connections for Modeling and Control of Gas-Lifted Oil Wells},
  author  = {Kittelsen, Jonas Ekeland and Antonelo, Eric Aislan and Camponogara, Eduardo and Imsland, Lars Struen},
  journal = {Applied Soft Computing},
  volume  = {158},
  pages   = {111603},
  year    = {2024},
  note    = {arXiv:2403.02289},
  url     = {https://ar5iv.labs.arxiv.org/html/2403.02289}
}
```

## Setup

On this machine use `/usr/local/bin/python3` (3.12) and `/usr/local/bin/pip3`. PyTorch 2.10 is already installed. Training is forced onto CPU.

```bash
cd /Users/rohit/gas_pinc
/usr/local/bin/pip3 install -r requirements.txt
export MPLBACKEND=Agg
```

Checkpoints are `model.pt`. Each PINC checkpoint stores `state_dict`, `y_min`, `y_max`, `T`, the well id, the architecture, `lambda_y`, `lambda_F`, and `friction="cubic"`.

## Run order

1. Build the feasible domain and run the steady-state gate. This writes `dataset/well1_domain.npz` and prints settled bottom-hole pressures in bar.

```bash
/usr/local/bin/python3 main_build_dataset.py --well 1
```

The gate requires at least one settled pressure in 80–120 bar (`P_res = 160 bar`). If it fails, the script exits with status 2 and does not justify a long train. Fix `P_0`, `D_bh`, or the pressure unit before training.

2. Adam, 1000 epochs, learning rate `7e-3`, `Nt = 1000`, `Nf = 10000`, seed 0. Writes `runs/well1/adam/model.pt`.

```bash
/usr/local/bin/python3 main_train_adam.py --well 1
```

A short run is `--epochs 2`. The default stays 1000.

3. L-BFGS from that checkpoint, strong Wolfe, history size 50, gradient tolerance `1e-9`, cap 1000 iterations. Writes `runs/well1/lbfgs/model.pt`.

```bash
/usr/local/bin/python3 main_train_lbfgs.py --well 1
```

A short run is `--iterations 2`. The default stays 1000. `--resume` continues `runs/well1/lbfgs/model.pt`.

4. Algebraic network, 4 hidden layers of 30 tanh units, 50000 training rows, 10000 validation rows, L-BFGS cap 5000. Writes `runs/well1/algebraic/model.pt`.

```bash
/usr/local/bin/python3 main_train_algebraic.py --well 1
```

5. Self-loop forecast, 50 steps of 60 s. Writes `forecast.npz` and `forecast.png`.

```bash
/usr/local/bin/python3 main_forecast.py --well 1
```

6. MPC. The default is a short smoke (3 plant steps, horizon 5). `--full` is the paper's 90 min loop with `N = 50` and `Nu = 45`.

```bash
/usr/local/bin/python3 main_mpc.py --well 1
/usr/local/bin/python3 main_mpc.py --well 1 --full --controller both
/usr/local/bin/python3 main_mpc.py --well 1 --noise 0.05 --controller both
```

```bash
/usr/local/bin/python3 -m pytest
```

The paper's well-1 numbers (training loss `7.48e-7`, validation loss `1.43e-6`, state IAE `0.00994`, noiseless MPC IAE `0.038 bar`) are printed next to a run as orientation. They are not a pass/fail check. Scaling bounds are discovered here and will not match the authors' unpublished bounds. Do not treat a smoke run as those figures.

The well-1 gate was run. Settled or full-horizon bottom-hole pressures on the valve grid from `(3500, 300, 8000)` kg fall in about `76–105 bar`, including `82.8, 84.2, 86.6, 101.9, 102.2, 103.0, 105.2`. That passes the `80–120 bar` gate. The saved cube is approximately

```
m_G_an in [2332, 3985] kg
m_G_tb in [41, 439] kg
m_L_tb in [4772, 11410] kg
```

`runs/well1/adam` and `runs/well1/lbfgs` currently hold a 2-epoch / 2-iteration smoke (seed 0, loss weights 1 and 1), not the 1000-epoch configuration. A full Adam + L-BFGS train is implemented and is what the commands without `--epochs` / `--iterations` run. It has not been executed, so the paper losses and the `0.038 bar` IAE are not results of this repository.

## States, inputs, and scaling

Differential state, kilograms: `y = [m_G_an, m_G_tb, m_L_tb]`.

Valves, held constant on each interval of `T = 60 s`: `u = [u1, u2]` in `[0, 1]`. `u1` is the production choke (18f). `u2` is the gas-lift choke (18a).

PINC input: `X = [t / T, y0 scaled to [0, 1], u]`. Output: scaled state at time `t`. A control step always evaluates the net at `t = T`. A long rollout re-injects that prediction as the next initial state.

Algebraic outputs: `z = [P_bh (bar), w_G_in, w_G_res, w_L_res]`. The network sees five inputs, the three masses scaled with the same bounds plus both valves. Reported pressures are bar; the ODE keeps pascal.

Physics residual, per state, in scaled units:

```
dy_s/dt_s = d(network)/d(t/T)
F = dy_s/dt_s - ode_torch(y, u, friction="cubic") * T / (y_max - y_min)
```

`MSE_F` is the mean of `F**2` over the collocation batch and the three states. `MSE_y` matches the network at `t = 0` to the scaled initial state. `loss = lambda_y * MSE_y + lambda_F * MSE_F` with both weights equal to 1. `y_min`, `y_max`, and `T` are not optimized.

Validation does not use the physics loss. It is the mean squared error, in scaled units, between the network at `t = T` and a 60 s Haaland RK45 step, on up to 100 feasible pairs.

## Decisions the paper leaves open

These are fixed in this code. They are not in the HTML paper.

- `P_0 = 1.01325 bar` (`1.01325e5 Pa`) in (18f). Table 4 does not list it.
- `D_bh = sqrt(4 S_bh / pi) = 0.19995 m` for every well. Table 4 does not list it. `F_bh` is constant, so this choice is a constant pressure offset (about `0.019 bar` on well 1) and is kept.
- The mass fraction in (18d) and (18e) is (21b), `alpha_G_bh_m = GOR / (GOR + 1)`. It is constant. Wells 1 and 2 have `GOR = 0`, so the reservoir stream is liquid.
- Symbols `T_a` and `V_a` in (19a) are `T_an` and `V_an`.
- The algebraic net input is 5-D. Section 4.6 says 3-D (the states only). Section 4.5 and Appendix C pass the state and the control into `F_x_to_P_bh`. `P_bh` depends on `u2` through `w_G_in` and the tubing friction, and `w_G_in` depends on `u2` directly. `u1` does not appear in `P_bh`; it is still an input so one net matches the CasADi call `F_x_to_P_bh(x, u)`.
- The state box is the min/max of Haaland RK45 trajectories on the valve grid `{0.2, 0.4, 0.6, 0.8, 1.0}`, expanded by 5% on each side, then scaled to `[0, 1]`. The paper does not print the bounds.
- The prompt's well-1 guess `(2000, 300, 8000)` kg at `u = (0.5, 0.5)` makes the extrapolated tubing-top mixture density negative (`alpha_L_tb_t < 0` while `rho_G_tb_t` is small), so the explicit ODE is undefined there. Feasible states with `m_G_an = 2000` kg that were found on a grid sit near `128–160 bar`, above the MPC band. The sweep starts from `(3500, 300, 8000)` kg, which is feasible and gives `P_bh ≈ 94.7 bar` at `u = (0.5, 0.5)`.
- The physics residual is in scaled units, as written above. Masses are thousands of kilograms; an unscaled residual would not sit near the paper's `1e-6` losses.
- `lambda_y = lambda_F = 1`. Change them only if a real training run shows one term at least 100 times the other after the network is producing feasible states, and write the new values into the checkpoint and this file.
- The plant and the validation reference use Haaland friction. The PINC physics loss uses the Appendix B cubic after clipping Reynolds number to the well's fit interval. If that cubic is negative, `lambda` is floored at `1e-6` and the hit is counted. On the tabulated intervals the cubics stay positive (well 1 is about `0.028` at `Re = 13000` and about `0.018` at `Re = 115000`), so the floor is a guard, not a fit.
- `sqrt(max(x, 1e-3))` is used only inside `ode_torch(..., friction="cubic")`, on (18a), (18b), and (18f). The RK45 plant uses `max(., 0)` and a plain square root, and it raises `InfeasibleState` on a non-finite result. It does not clamp and it does not return zeros.
- Seed 0 for the default dataset, Adam, L-BFGS, algebraic, forecast, and MPC runs.
- Each MPC reference level is held for 30 min: 95 bar, then 97 bar, then 99 bar. The paper's figure is three steps of 2 bar from 95 to 99. The dwell length is not in the text.
- Float64 for the ODE, the networks, and CasADi. CPU even if MPS is available.
- Velocities (22a)–(22d) use the constant `w_res_bar`, not the instantaneous reservoir flow. (22b) uses `w_G_in`, not `w_G_inj`.
- `F_bh`, `Re_bh`, and `lambda_bh` are computed once per well with Haaland. They do not depend on the state.
- A network prediction that leaves the explicit model is not passed to the ODE during training. Those collocation rows contribute `(y_s - 0.5)^2` so the loss stays finite and the weights still move toward the middle of the scaled box. The ODE functions themselves still raise `InfeasibleState`. After training predictions stay feasible, this term is inactive and the loss is the paper's weighted sum.
- `P_bh` in the NLP is an expression, not an `opti.subject_to` equality. Appendix C is explicit that a constraint on `P_bh` only adds cost. CasADi cannot assign into an `opti.variable` slice, so `P_bh` is not a free decision variable.
- The first MPC solve sets the first `R` block to zero (`R = diag(1e3, 1e3)` on later moves). That is the paper's option when the previous valve position is not a steady-state input. Later solves warm-start by shifting the previous solution one step.
- `Q` has `N` diagonal weights: 1 on the first `Nu` outputs and 100 on the last `N - Nu`. For the paper's horizon that is 45 ones and 5 hundreds. The CasADi listing sizes `Q` as `(N+1)` by `(N+1)`; the sum in (11a) has `N` terms, and that is what is implemented.
- State IAE averages the `C` knots after each self-loop step. The shared initial state is stored in the trajectory and is not inside the average. `C = 50` for the default forecast.
- SLMPC linearizes the Haaland torch twin with `torch.autograd` at the measured state and passes numeric `A, B, delta, C, D` into CasADi. It does not finite-difference through PyTorch inside IPOPT.
- The full 90 min, `N = 50` IPOPT loop is slow. `main_mpc.py` without `--full` builds the NLP and takes a few plant steps (default horizon 5, 3 minutes) so the smoke is runnable.

## Layout

```
gas_pinc/
  readme.md
  requirements.txt
  data/
    param.py            three wells, units, cubic coefficients
    well_ode.py         explicit ODE, numpy and torch
    friction.py         Haaland and the cubic clamp
    skip_pinc.py        Wang 2021 encoder skip network
    algebraic_nn.py     feedforward algebraic net
    sample.py           domain discovery and collocation sets
    pinn_well.py        Adam then L-BFGS
    forecast.py         self-loop versus RK45
    mpc.py              CasADi PINC-MPC and SLMPC
    LossHistory.py
    TrainState.py
    Logger.py
    utils.py            checkpoints, plots, seed
  main_build_dataset.py
  main_train_adam.py
  main_train_lbfgs.py
  main_train_algebraic.py
  main_forecast.py
  main_mpc.py
  tests/
```

`dataset/well{1,2,3}_domain.npz` is generated. Nothing is downloaded.

## Architecture

Default PINC: 6 hidden layers, 30 neurons, tanh, two encoders of width 30, linear output. Glorot uniform weights, zero biases. `--no-skip` is the same depth and width without the encoder gates, for the ablation only. The constructor takes `n_layers` and `n_neurons`. The paper's search set was `{2, 4, 6, 8, 10}` layers and `{10, 20, 30, 40}` neurons. That search is not run by default.

Default algebraic net: 4 hidden layers, 30 tanh neurons, linear output, trained by L-BFGS on Haaland algebraic targets. No physics loss. Outputs are standardized with the training mean and standard deviation and inverted before reporting. A zero standard deviation (well 1 has `w_G_res = 0`) is floored at `1e-8`.
