# Dragging a Brownian Particle: Learning the Optimal Protocol

## The Problem

An optical tweezer holds a colloidal bead in a harmonic trap. You move the trap center
from $0$ to $\Delta\lambda$ in a finite time $t_f$, and the bead follows through a viscous
fluid. Moving a harmonic trap does not change its free energy, so $\Delta F = 0$ and every
bit of work you do is dissipated. The question is which protocol $\lambda(t)$ costs the
least mean work.

The obvious guess is to drag at constant speed. It is wrong. Schmiedl and Seifert,
PRL 98, 108301 (2007), showed that the optimal protocol **jumps** at both ends: the trap
leaps ahead at $t = 0$, moves linearly, and leaps again at $t_f$. A reinforcement-learning
agent that is told nothing except the work it pays on noisy trajectories should find
those jumps. Whitelam, PRX 13, 021005 (2023), did this with neuroevolution.

This session also asks what the agent gains from *looking*. If the agent measures the
bead's position before each move, the policy becomes a genuine function of the state,
and the mean work drops below the open-loop optimum. It can even go negative, which
makes the setup an information engine. Brownian information engines of this kind have
been built in the lab, for example Paneru et al., PRL 120, 020601 (2018).

The model has a linear–Gaussian environment and a quadratic cost, so the RL problem has an
exact solution at every level. The open-loop optimum comes from a linear solve, and the
feedback optimum comes from a backward Bellman recursion. Every number RL produces in this
session gets compared with an exact one.

## Goal

1. Build the environment, an MDP in which every term has an exact counterpart.
2. Find the exact open-loop optimum and show that it converges to Schmiedl–Seifert.
3. Let policy gradient (REINFORCE) rediscover the jumps from noisy work samples.
4. Solve the Bellman equation exactly for the feedback problem, then compare two RL
   methods against it: policy gradient and fitted Q-iteration.

The point is to see what the state space and the action space buy you, and to check that
RL lands where Bellman says it should.

## The Setup

**Units.** Trap stiffness $k = 1$, friction $\gamma = 1$, $k_BT = 1$. Time is measured in
the relaxation time $\gamma/k$, length in $\sqrt{k_BT/k}$, and energy in $k_BT$. Use
$\Delta\lambda = 5$ and $t_f = 2$ throughout unless stated otherwise. Both are big enough
that the jumps are obvious and the thermal noise still matters.

**Dynamics.** The bead obeys the overdamped Langevin equation
$\dot x = -(x - \lambda) + \sqrt{2}\,\xi(t)$ in the potential
$U(x, \lambda) = \tfrac12 (x - \lambda)^2$. It starts in equilibrium,
$x_0 \sim \mathcal N(0, 1)$, with $\lambda_0 = 0$.

**Discretization: piecewise-constant protocols.** Split $[0, t_f]$ into $N$ intervals of
length $\Delta t = t_f/N$, and use $N = 20$ by default.

- At the start of interval $k = 1, \dots, N$ the trap jumps to $\lambda_k$ and stays there.
  At $t_f$ it jumps to $\lambda_{N+1} = \Delta\lambda$, which is fixed.
- While $\lambda$ is held constant the dynamics is an Ornstein–Uhlenbeck process, so
  integrate it **exactly**:
  $x_k = \lambda_k + (x_{k-1} - \lambda_k)\,a + \sqrt{1 - a^2}\,\xi_k$ with
  $a = e^{-\Delta t}$. This scheme has no time-step error. The only discretization is in
  the protocol.
- Work is done only when the trap moves, so a jump at fixed $x$ costs
  $w_k = U(x_{k-1}, \lambda_k) - U(x_{k-1}, \lambda_{k-1})$. Then
  $W = \sum_{k=1}^{N+1} w_k$.

**The MDP.** Write this out explicitly in the notebook, because it is the point of the
week.

| | open loop (steps 2–3) | feedback (steps 4–5) |
|---|---|---|
| state $s_k$ | the step index $k$ alone | $(k,\ x_{k-1},\ \lambda_{k-1})$, measured before acting |
| action $a_k$ | $\lambda_k \in \mathbb R$ | $\lambda_k \in \mathbb R$ |
| transition | the exact OU step | the exact OU step |
| reward | $r_k = -w_k$ | $r_k = -w_k$ |
| episode | $N$ actions, plus the forced final jump | same |

The open-loop agent never sees $x$. Its dynamics is stochastic, but it starts from a fixed
distribution and has nothing to condition on, so its optimal policy is a fixed sequence.
The feedback agent sees $x$, and that is where the state space earns its keep.

**Exact facts to use and to CHECK.**

- **Mean work.** The variance of $x$ stays at 1 for every protocol, because it starts in
  equilibrium and moving the center does not change the variance. The mean obeys
  $m_k = a\,m_{k-1} + (1 - a)\lambda_k$, and
  $\langle w_k\rangle = \tfrac12 (m_{k-1} - \lambda_k)^2 - \tfrac12 (m_{k-1} - \lambda_{k-1})^2$.
  So $\langle W\rangle$ is an exactly computable **quadratic** function of
  $u = (\lambda_1, \dots, \lambda_N)$.
- **Continuum optimum.** $W^* = \Delta\lambda^2/(t_f + 2)$. The protocol jumps to
  $c = \Delta\lambda/(t_f + 2)$ at $t = 0^+$, rises with slope $c$ to $c(t_f + 1)$, and jumps
  to $\Delta\lambda$ at $t_f$. Along this protocol the bead moves at constant speed $c$.
  Derive it in a markdown cell: write $\lambda = m + \dot m$ and the work becomes
  $\int \dot m^2\,dt$ plus boundary terms.
- **Constant-speed protocol.** $W_{\rm lin} = \frac{\Delta\lambda^2}{t_f}\Big[1 - \frac{1 - e^{-t_f}}{t_f}\Big]$.
  At $t_f = 2$ the optimum saves 11.9%.
- **Gaussian work.** For any open-loop protocol, $W$ is linear in the jointly Gaussian
  $x_k$, so it is Gaussian. Jarzynski's equality $\langle e^{-W}\rangle = e^{-\Delta F} = 1$
  then forces $\mathrm{Var}\,W = 2\langle W\rangle$.

**Reference values.** The numbers in this brief come from a scratch smoke test run before
the session. They are there to check against, not to copy.

## Natural Steps

1. **Environment and its exact checks.**
   - Write a vectorized episode runner. It takes a policy $(k, x, \lambda_{\rm prev}) \mapsto
     \lambda_k$ and $M$ trajectories, and returns the per-step work $(M, N+1)$ together with
     the states and actions. It must be vectorized over trajectories, and it may loop over
     the $N$ steps.
   - CHECK the simulated $\langle W\rangle$ against the exact mean-work formula, within the
     Monte Carlo error, for three protocols: the linear ramp, the exact optimum from
     step 2, and a random protocol. Ours used $M = 2\times10^5$ and the linear ramp
     sampled at interval midpoints, $\lambda_k = \Delta\lambda (k - \tfrac12)/N$:

     | protocol | simulated $\langle W\rangle$ | exact |
     |---|---|---|
     | linear | 7.0970 ± 0.0084 | 7.0973 |
     | optimal | 6.2562 ± 0.0079 | 6.2526 |

     $\mathrm{Var}\,W / 2\langle W\rangle$ came out at 0.996, 1.001 and 0.998 for the three
     protocols. CHECK this ratio against 1 using a bootstrap error bar.
   - **Jarzynski, and why it can fail.** CHECK $\langle e^{-W}\rangle = 1$ at
     $\Delta\lambda = 1$, using that $\Delta\lambda$'s optimal protocol. Then show what
     happens as $\Delta\lambda$ grows. For Gaussian work the per-sample relative standard
     deviation of $e^{-W}$ is $\sqrt{e^{2\langle W\rangle} - 1}$, so the estimator is
     dominated by rare trajectories with $W < 0$. Ours, at $M = 2\times10^5$:

     | $\Delta\lambda$ | $\langle W\rangle$ | $\langle e^{-W}\rangle$ | $\sqrt{e^{2\langle W\rangle}-1}$ |
     |---|---|---|---|
     | 0.5 | 0.063 | 0.9990 ± 0.0008 | 0.36 |
     | 1 | 0.252 | 0.9976 ± 0.0018 | 0.81 |
     | 2 | 1.003 | 0.9931 ± 0.0055 | 2.5 |
     | 3 | 2.255 | 0.9845 ± 0.0174 | 9.4 |
     | 5 | 6.260 | 0.8775 ± 0.0971 | 518 |

     At $\Delta\lambda = 5$ the estimate is low, and the quoted error bar itself is
     meaningless because it is computed from the same undersampled tail. Say so in the
     notebook.

2. **The exact open-loop optimum (and the headline figure).**
   - Write $\langle W\rangle(u) = \tfrac12 u^T H u - g^T u + \text{const}$ by building the
     affine maps $u \mapsto m$ and $u \mapsto \lambda$ as matrices, then solve $Hu^* = g$.
   - CHECK that $H$ is positive definite. Its eigenvalues at $N = 20$ run from 0.0851 to
     0.20. The landscape is shallow everywhere, and that matters in step 3.
   - CHECK that the increments of $u^*$ are constant to $10^{-12}$. Ours were constant to
     $4\times10^{-15}$, with increment 0.124948, against $c\,\Delta t = 0.125$. The
     discrete optimum is exactly a ramp plus two jumps.
   - CHECK convergence to Schmiedl–Seifert. $W^*_N \ge W^*$, because a piecewise-constant
     protocol is still a physical protocol, and the excess falls as $N^{-2}$:

     | $N$ | 5 | 10 | 20 | 40 | 80 |
     |---|---|---|---|---|---|
     | $W^*_N - W^*$ | $4.13\times10^{-2}$ | $1.04\times10^{-2}$ | $2.60\times10^{-3}$ | $6.51\times10^{-4}$ | $1.63\times10^{-4}$ |

     So the excess is $\approx 1.04/N^2$. At $N = 20$: $W^*_N = 6.252603$,
     $u_1 = 1.31299$, $u_N = 3.68701$. The continuum values are 1.25 and 3.75, offset by half
     a step of the ramp.
   - **Headline figure.** Top panel: $\lambda(t)$ as step plots for the linear ramp, the
     exact $u^*$, the RL protocol from step 3, and the mean protocol of the exact feedback
     policy from step 4, with the continuum Schmiedl–Seifert curve overlaid. Bottom panel:
     learning curves, meaning the exact mean work of the current policy against the
     iteration, for open-loop and feedback RL. Draw horizontal lines at $W_{\rm lin}$,
     $W^*_N$ and $W_{\rm fb}$. Fill in each piece as its step lands.

3. **REINFORCE rediscovers the jumps (open loop).**
   - Use the Gaussian policy $\lambda_k \sim \mathcal N(\theta_k, \sigma^2)$ with
     $\sigma = 0.3$, starting from the linear ramp. The agent sees only per-step rewards
     from single trajectories.
   - The gradient estimator is
     $\hat g_k = \big\langle (G_k - b_k)(\lambda_k - \theta_k) \big\rangle / \sigma^2$, where
     $G_k = -\sum_{j \ge k} w_j$ is the **reward-to-go** and $b_k$ is the batch mean of
     $G_k$.
   - **Predict before you run it.** Gaussian smoothing of a quadratic adds only a constant,
     so $\mathbb E[\hat g] = -(H\theta - g)$ exactly, and the fixed point is $u^*$ itself.
     CHECK this. Average $\hat g$ over 20 batches of 20000 episodes at the linear ramp and
     compare with $-(H\theta - g)$ component by component. Ours gave $\chi^2/\text{dof} = 1.25$
     over 20 components. Using the full return in place of the reward-to-go is also
     unbiased, but it more than doubled the per-episode variance (162 against 74). Show
     both.
   - **Training.** Use Adam with learning rate 0.02, $B = 1000$ episodes per update, 3000
     updates, and Polyak averaging of $\theta$ over the second half, for 5 seeds. Plot the
     learning curve with the *exact* mean work of $\theta$, which is free and has no noise.
     Ours took 9 s for all five seeds:
     - Last iterate: excess $W(\theta) - W^*_N$ between 0.016 and 0.046.
     - Polyak average: excess between 0.0022 and 0.0035. The gap from the linear ramp to the
       optimum is 0.845.
     - $\theta_1$ between 1.26 and 1.41 (exact 1.313), and $\theta_N$ between 3.68 and 3.74
       (exact 3.687). Both jumps were found from a start of 0.125 and 4.875.
   - CHECK that the Polyak excess is below 1% of the linear-to-optimal gap for every seed,
     and that $\theta_1$ and $\theta_N$ land within 0.15 of $u^*$. This is a convergence
     criterion for an optimizer, not a sampling estimate, so the natural scale is the gap
     rather than an error bar.
   - **Be honest about the protocol error.** In several components the seed-averaged
     protocol sat 0.01 to 0.04 above $u^*$, beyond the seed-to-seed standard error.
     - The Hessian eigenvalues all lie between 0.085 and 0.20, so the condition number is
       only about 2.3. The problem is not one flat direction: the whole landscape is shallow.
     - A residual of a few hundredths pulls back with a gradient of order $10^{-2}$, while a
       batch gradient carries noise of order 0.3.
     - RL finds near-optimal *work* well before it pins down the optimal *protocol*. Show it:
       project $\theta - u^*$ onto the eigenvectors of $H$.

4. **Feedback: exact Bellman, then policy gradient.**
   - Augment the state as $z = (x, \lambda_{\rm prev}, 1)$. The value function is exactly
     quadratic, $V_k(z) = \tfrac12 z^T P_k z + c_k$. Start the backward recursion from the
     forced final jump,
     $V_N(z) = \tfrac12 (x - \Delta\lambda)^2 - \tfrac12 (x - \lambda_{\rm prev})^2$, and step
     $V_{k-1}(z) = \min_u \big[ \tfrac12 (x - u)^2 - \tfrac12 (x - \lambda_{\rm prev})^2
     + \mathbb E\, V_k(z') \big]$. The noise adds $\tfrac12 (1 - a^2) (P_k)_{xx}$ to the
     constant. The minimizer is linear, $u^* = g_k x + b_k \lambda_{\rm prev} + h_k$. The
     expected optimal work is $\mathbb E_{x_0} V_0(x_0, 0)$.
   - The state cost $-\tfrac12 (x - \lambda_{\rm prev})^2$ is not convex. CHECK that the
     coefficient $K_{uu}$ of $u^2$ is positive at every step, so that the minimum exists.
     Ours had a minimum of 0.186 at $N = 20$.
   - **Exact policy evaluation.** Under any linear policy $(x, \lambda)$ stays Gaussian, so
     propagate its mean and covariance to get the exact expected work. This is RL's
     "policy evaluation", done exactly, and it gives every learning curve in this step.
     CHECK that it reproduces the Bellman value to $10^{-10}$.
   - **Reference values at $N = 20$.**
     - $W_{\rm fb} = 5.46970$, against $W^*_N = 6.25260$.
     - The gain $g_k$ runs from 0.7374 down to 0.5.
     - At the last step $\lambda_N = (x + \Delta\lambda)/2$, halfway between the bead and the
       target.
     - Simulating the exact policy with $M = 2\times10^5$ gave $5.4594 \pm 0.0082$, and
       $P(W < 0) = 0.046$.
   - **Three structural CHECKs.**
     1. The coefficient on $\lambda_{\rm prev}$ is exactly 0, so the decision needs only
        $(k, x)$, even though the value function depends on $\lambda_{\rm prev}$. Explain why.
     2. The feedback law applied along the *mean* open-loop path reproduces $u^*$ to
        $10^{-12}$. Ours agreed to $1.3\times10^{-14}$. This is certainty equivalence.
     3. The simulated work agrees with Bellman within the Monte Carlo error.
   - **The value of information** is $W^*_N - W_{\rm fb}$, and it grows with $N$ as you
     measure more often:

     | $N$ | 10 | 20 | 40 | 80 | 160 | 320 |
     |---|---|---|---|---|---|---|
     | $W_{\rm fb}$ | 5.5482 | 5.4697 | 5.4287 | 5.4077 | 5.3970 | 5.3917 |
     | $W^*_N - W_{\rm fb}$ | 0.712 | 0.783 | 0.822 | 0.843 | 0.853 | 0.858 |

   - **Policy gradient with feedback.**
     - Use a linear Gaussian policy with mean $g_k x + b_k \lambda_{\rm prev} + h_k$,
       $\sigma = 0.3$, and the same REINFORCE estimator with features $(x, \lambda_{\rm prev}, 1)$.
     - Initialize at the open-loop optimum, $g = b = 0$ and $h = u^*$.
     - Train with Adam at learning rate 0.01, $B = 1000$, 6000 updates, and Polyak averaging
       over the second half. Ours took about 4 s per seed.
     - Over 4 seeds:
       - $W$ landed between 5.4793 and 5.4801, which is 98.7–98.8% of the feedback gain.
       - $\max_k |g_k - g_k^*|$ was 0.10–0.18.
       - $b_k$ drifted to about $-0.2$ in every seed instead of 0. One hypothesis: early on,
         when $g \approx 0$, $\lambda_{\rm prev}$ acts as a proxy for $x$, and the $b$
         direction is nearly flat, so it relaxes back slowly. Test the hypothesis, for
         example by training longer or by initializing differently, rather than asserting it.
     - CHECK that each seed captures at least 95% of the exact feedback gain.

5. **Value-based RL: fitted Q-iteration on the state–action space.**
   - $Q_k(x, \lambda_{\rm prev}, u)$ is exactly quadratic. Fit it backward in $k$ by least
     squares on the 10 monomials of degree at most 2 in $(x, \lambda_{\rm prev}, u)$.
     - The regression target is $w + V_{k+1}(x', u)$, where $V_{k+1}$ is the minimum over
       $u'$ of the fitted $Q_{k+1}$, or the exact terminal value at the last step.
     - Draw states and actions uniformly from a box that covers the optimal path. Ours used
       $x \in [-3, 8]$ and $\lambda_{\rm prev}, u \in [-2, 7]$, with one OU step per sample.
   - The greedy policy solves $\partial Q_k/\partial u = 0$, which is linear in the state. Compare its
     coefficients with $(g_k, b_k, h_k)$ from Bellman, and its exact work with $W_{\rm fb}$.
     Ours:

     | samples per step | $W$ | $\max\lvert g - g^*\rvert$ | $\max\lvert b\rvert$ | $\max\lvert h - h^*\rvert$ |
     |---|---|---|---|---|
     | $10^4$ | 5.4751 | 0.040 | 0.036 | 0.118 |
     | $10^5$ | 5.4709 | 0.023 | 0.007 | 0.078 |
     | Bellman | 5.4697 | 0 | 0 | 0 |

     The $10^5$ case took 0.3 s. It found $b \approx 0$, which policy gradient did not.
   - Close the step with the comparison. Fitted Q-iteration wins here because its function
     class contains the exact $Q$. That is knowledge the policy-gradient agent was never
     given. Say what would break if the environment were not linear–Gaussian.

## Push Harder

Predict first for each item, then measure.

- **Information engine.** Scan $t_f$ at $N = 20$ and $\Delta\lambda = 5$. Ours:

  | $t_f$ | 0.5 | 1 | 2 | 4 | 8 |
  |---|---|---|---|---|---|
  | $W^*$ | 10.000 | 8.333 | 6.250 | 4.167 | 2.500 |
  | $W_{\rm lin}$ | 10.653 | 9.197 | 7.096 | 4.716 | 2.735 |
  | $W_{\rm fb}$ | 9.850 | 7.995 | 5.470 | 2.416 | −1.035 |

  At $t_f = 8$ the feedback agent does *negative* net work: it turns heat from the bath into
  work by using its measurements. Find the $t_f$ where $W_{\rm fb}$ crosses zero (ours: 6.54),
  then predict the $t_f \to \infty$ limit at fixed $N$ before computing it.
  - Our naive prediction was $-N/2$, half a $k_BT$ per measurement. It missed: the exact
    value is $-8.08$. A trap that follows the bead inherits the bead's random walk, and it
    must still end at $\Delta\lambda$.
  - Solving the fully relaxed problem by hand gives
    $-\frac{N+1}{2} + \frac{\Delta\lambda^2 + 1}{2(N+1)} + \frac12 H_N$, where $H_N$ is the
    harmonic number. CHECK it against Bellman.
- **Continuum limit of feedback.** $W_{\rm fb}(N)$ above converges like $1/N$ toward about
  5.386, while $W^*_N$ converges like $1/N^2$. Extrapolate, and explain why the two rates
  differ.
- **Noisy measurement (a POMDP).** Measure $y = x + \eta$ with
  $\eta \sim \mathcal N(0, s^2)$.
  - The state is no longer observed. Predict the optimal policy: a Kalman filter for $x$
    combined with the same gains, by the separation principle.
  - Predict also that $W$ interpolates between $W_{\rm fb}$ at $s = 0$ and $W^*_N$ as
    $s \to \infty$.
  - Then let REINFORCE learn on raw $y$, and compare.
- **Stiffness protocol.** Change the stiffness from 1 to 5 at a fixed center. Schmiedl and
  Seifert solved this case too, and it also has jumps.
  - Now $\langle W\rangle$ follows from a deterministic recursion for $\langle x^2\rangle$ and
    is no longer quadratic. The exact discrete optimum comes from a deterministic
    minimization, with no sampling.
  - Treat this as exploratory, because we did not smoke-test it.
- **Tabular Q-learning.** This gives the textbook picture of state and action spaces, at a
  cost you can measure.
  - **Make the spaces finite.** Put the measured $x$ on a grid of spacing $\delta x$ over
    $[-3.5, 8]$ and $u$ on a grid of spacing $\delta u$ over $[-1, 6.5]$.
  - **Regroup the rewards** as $-\tfrac12(x - u)^2 + \tfrac12(x' - u)^2$, plus the constant
    $-\tfrac12 x_0^2$ and the final $\tfrac12(x_N - \Delta\lambda)^2$, so that the state needs no
    $\lambda_{\rm prev}$.
  - **The environment must *be* the discrete MDP** for $Q^*$ to be exact. Snap $x'$ to the
    nearest grid point, use exact Gaussian bin masses for $P(x'|x,u)$, and get $Q^*$ by value
    iteration.
  - **Measure the discretization cost** by running the tabular policy in the continuous
    environment, paired on the same random numbers with the exact feedback policy. The
    error bar drops from about 0.008 to below 0.001.
  - Ours:

    | $\delta x = \delta u$ | table entries | $V^*$ of the grid MDP | cost vs $W_{\rm fb}$ (paired) |
    |---|---|---|---|
    | 0.5 | 7,680 | 5.4403 | 0.0555 ± 0.0007 |
    | 0.25 | 29,140 | 5.4607 | 0.0143 ± 0.0004 |
    | 0.125 | 113,460 | 5.4658 | 0.0041 ± 0.0002 |

    The grid MDP's own optimum sits *below* $W_{\rm fb}$, because snapping makes it a different
    system.
  - **Q-learning run.** Watkins Q-learning on the $\delta = 0.25$ grid, with $\epsilon = 0.2$ and
    step size $1/n^{0.7}$. Run $B = 2000$ episodes in parallel and average the samples that hit
    the same cell. Start at $Q = 0$, and take the greedy policy over actions actually tried.
  - **Our result.** 6M episodes ($1.2\times10^8$ transitions, 28 s) reached $W = 5.521$ in the
    grid MDP. That is 92% of the gain over open loop, with the greedy action within one grid
    step of optimal on 87% of visited states. In the continuous environment the policy cost
    $0.073$ more than $W_{\rm fb}$.
  - Two earlier attempts went wrong:
    - With uniformly random actions, and a greedy evaluation that also counted never-tried
      actions (still at their initial $Q = 0$), the policy was a disaster (11.2).
    - A step size of $1/n$ stalled near 8.
  - Compare with fitted Q-iteration's 99.8% from $2\times10^6$ transitions: the
    difference is generalization.

## Out of Scope

- Underdamped dynamics. The optimal protocol there has δ-function peaks (Gomez-Marin,
  Schmiedl and Seifert, J. Chem. Phys. 129, 024114 (2008)). Interesting, but it belongs to
  a different session.
- Neural-network policies. Every optimal policy in this session is linear in the state,
  and RL is judged against that exact answer. A neural network would only blur the
  comparison.
- Sagawa–Ueda bounds with perfect measurement. The mutual information is infinite for a
  noiseless continuous measurement. That bound only becomes interesting with the noisy
  measurement in Push Harder.
- Gym, stable-baselines or any RL library. Write the environment and the agents yourself.
  Each of them is a few dozen lines.

## Deliverables and Constraints

- **Layout.** Follow the previous sessions' layout:
  - `trap.py` holds the environment and the exact side. That covers the exact OU step, the
    vectorized episode runner, the exact mean work, the quadratic form $(H, g)$ and $u^*$,
    the continuum formulas ($W^*$, the continuum protocol, $W_{\rm lin}$), the Bellman
    recursion, and the exact linear-policy evaluator by moment propagation.
  - `agents.py` holds REINFORCE, open-loop and feedback, with Adam and Polyak averaging,
    plus fitted Q-iteration. Keep it in plain numpy, a few dozen lines each.
  - Fast checks go in a pytest `test_trap.py` that runs well under a minute:
    - $u^*$ has constant increments, and $W^*_N$ approaches $W^*$ as $N^{-2}$;
    - the simulated mean work agrees with the exact value within the Monte Carlo error;
    - $\mathrm{Var}\,W = 2\langle W\rangle$;
    - the Bellman value equals the moment-propagation value;
    - certainty equivalence holds;
    - the REINFORCE gradient is unbiased against $-(H\theta - g)$.
  - `build_nb.py` writes and executes `Optimal_Trap_Dragging.ipynb`. The printed
    `CHECK ... PASS/FAIL` lines live in the notebook.
  - A `.gitignore` in the folder tracks only the `.md` and the `.ipynb`, as in previous
    weeks.
- **Notebook order.**
  - Open with a markdown cell that writes out the MDP table: the state, action, transition
    and reward for both settings.
  - Then follow the order of the steps, with the headline figure filled in as results land.
  - A summary-of-checks cell always comes last.
- **Libraries.** numpy and matplotlib only. scipy is allowed for a cross-check in the tests.
- **Seeds.** Fix all seeds. Monte Carlo CHECKs compare against their own error bars.
  Optimizer-convergence CHECKs use the relative criteria stated above, and the notebook
  says which kind each CHECK is.
- **Run time.** The whole notebook should run in about two minutes on a laptop. Ours took
  75 s, including tabular Q-learning.
- **Build incrementally.** Grow `build_nb.py` one section at a time, and rebuild and execute
  the notebook as each section lands. The notebook on disk should always show every result
  finished so far.
- **Session length.** The session is meant to fit in about 100 minutes. Steps 1–4 are the
  core. Step 5 and Push Harder are where the remaining time goes.
