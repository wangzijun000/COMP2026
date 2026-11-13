# Rabi Oscillations and the Cramér–Rao Bound

## The Problem

The manual day trained networks to regress and classify, and judged them by a test loss.
A test loss tells you whether one network is better than another. It does not tell you
whether the network is *good*: how far it is from the best any method could possibly do.
For most problems that question has no answer. For parameter estimation from noisy counts
it has an exact one.

A two-level atom driven with Rabi frequency $\Omega$ at detuning $\Delta$, starting in the
ground state, is found excited at time $t$ with probability

$$P(t;\Omega,\Delta) = \frac{\Omega^2}{W^2}\sin^2\!\Big(\frac{W t}{2}\Big),
\qquad W = \sqrt{\Omega^2 + \Delta^2}.$$

An experiment measures at times $t_k$, $N$ shots each. Every shot projects the atom, so the
number of excited counts is $n_k \sim \text{Binomial}(N, P_k)$. That is quantum projection
noise, and it is the only noise in this problem.

From the counts $\{n_k\}$, estimate $\Omega$ and $\Delta$. The Fisher information of the
data is

$$I_{ij}(\theta) = \sum_k \frac{N}{P_k(1-P_k)}\,
\frac{\partial P_k}{\partial\theta_i}\frac{\partial P_k}{\partial\theta_j},$$

and the Cramér–Rao bound says that any **unbiased** estimator has covariance at least
$I^{-1}$. That is the yardstick for the network.

## Goal

Build three estimators of $(\Omega, \Delta)$ from the same simulated data: the bound
itself, a maximum-likelihood fit, and a neural network trained on simulated experiments.
Show that the MLE sits on the bound, measure how close the network gets, and explain
every place where an estimator appears to *beat* the bound.

The emphasis is on the yardstick. The physics supplies a forward model simple enough that
the best possible answer is known.

## The Setup

Use dimensionless units throughout and keep these fixed so that your numbers compare to
the references below:

- Measurement times $t_k = 0.25\,k$ for $k = 1, \dots, 24$, so $t_\max = 6$.
- $N = 100$ shots per time point.
- The prior, used to generate training data: $\Omega \sim U[1, 2]$, $\Delta \sim U[-1, 1]$.
  This gives one to two Rabi periods inside the window.

**The degeneracy.** $P$ depends on $\Delta$ only through $\Delta^2$. The sign of the
detuning is invisible to this measurement, and $\partial P/\partial\Delta = 0$ at
$\Delta = 0$, so the Fisher information for $\Delta$ vanishes there. Estimate
$u = \Delta^2$ instead. Its Fisher information stays finite at $u = 0$. Most of the
interesting behaviour in this session comes from this one fact.

**Derivatives, by hand.** With $A = \Omega^2/W^2$ and $s = \sin^2(Wt/2)$:
$\partial W/\partial\Omega = \Omega/W$, $\partial W/\partial u = 1/(2W)$,
$\partial A/\partial\Omega = 2\Omega u/W^4$, $\partial A/\partial u = -\Omega^2/W^4$, and
$\partial s/\partial W = \tfrac{t}{2}\sin(Wt)$. Check these against central finite
differences; the relative error should be $\lesssim 10^{-6}$. That is the first CHECK line in
the notebook.

## Natural Steps

1. **The forward model and the bound.**
   - Write $P$, its analytic gradient in $(\Omega, u)$, the Fisher matrix, and the CRB.
   - Reference values at $N = 100$, as standard deviations from $\sqrt{\mathrm{diag}\,I^{-1}}$:

     | $(\Omega, \Delta)$ | $\sigma_\Omega$ | $\sigma_\Delta$ | $\sigma_u$ |
     |---|---|---|---|
     | (1.5, 0.5)  | 0.01115 | 0.03085 | 0.03085 |
     | (1.5, 0.05) | 0.00662 | 0.13204 | 0.01320 |
     | (1.0, 0.8)  | 0.01556 | 0.01845 | 0.02952 |
     | (2.0, 0.5)  | 0.01357 | 0.04585 | 0.04585 |

     At $(1.5, 0.5)$, $\sigma_u = 2\Delta\,\sigma_\Delta$ happens to equal $\sigma_\Delta$,
     because $2\Delta = 1$. At $\Delta = 0$ exactly, $\sigma_u = 0.01240$ while
     $\sigma_\Delta$ is infinite. The $\Omega$–$u$ correlation at $(1.5, 0.5)$ is $-0.73$.
   - Make the point in the notebook that the bound scales as $1/\sqrt N$. Check it by
     computing it at two values of $N$; this is exact, not statistical.

2. **Headline figure: one experiment, three answers.**
   - Simulate one dataset at $(\Omega, \Delta) = (1.5, 0.5)$. Plot $n_k/N$ with binomial
     error bars $\sqrt{P(1-P)/N}$, the true curve, and the curves from the MLE and network
     estimates. The reader should see that the data are noisy and the fits are not.
   - This is the first figure in the notebook. Fill in the MLE and network curves as those
     steps land.

3. **Maximum likelihood.**
   - The log-likelihood is $\sum_k [\,n_k\ln P_k + (N - n_k)\ln(1 - P_k)\,]$. It is
     multimodal in $\Omega$, because a slightly different frequency can line up with a
     different set of fringes. Start with a coarse grid over $(\Omega, |\Delta|)$, then
     refine with Fisher scoring, $\theta \leftarrow \theta + I(\theta)^{-1}\nabla\ell$. Fisher
     scoring reuses the step-1 code, and that is the reason to use it.
   - Estimate $|\Delta|$; the sign cannot be recovered.
   - **Check at (1.5, 0.5).** Over 2000 simulated datasets, our MLE gave
     $\mathrm{std}(\hat\Omega) = 0.01110$ and $\mathrm{std}(|\hat\Delta|) = 0.03067$, with
     mean $(1.5004, 0.4979)$. That is on the bound to within the Monte Carlo error of a
     standard deviation from 2000 samples, about $1.6\%$. Print the ratio to the CRB with
     that error bar.
   - **Damp the scoring steps.** Halve each step until the log-likelihood does not decrease.
     Undamped, one dataset in tens of thousands overshoots from a correct grid start to
     $\Omega < 0$ (where $P$ is the same, since it depends on $\Omega^2$). Guard this with a CHECK
     that the MLE's likelihood is never below the likelihood at the truth.
   - **Constrain $u \ge 0$.** For $u < 0$ the amplitude $\Omega^2/W^2$ exceeds 1 and $P$ stops being
     a probability. Letting $\hat u$ go negative produces a spurious second peak in its
     histogram. Project to $u = 0$ and take a 1D step in $\Omega$ there.
   - **The first apparent violation, at (1.5, 0.05).** Here $\sigma_u = 0.0132$ is five times the
     true $u = 0.0025$, and about half the datasets put $\hat u$ exactly at 0. The MLE gives
     $\mathrm{rmse}(|\hat\Delta|) \approx 0.06$, half of the "bound" of $0.132$, and
     $\mathrm{rmse}(\hat u) \approx 0.011$, below $\sigma_u$ as well. Explain why in the notebook: any
     estimator that respects $u \ge 0$ is biased upward next to the boundary, in either
     parametrization, and the CRB only constrains unbiased estimators. The bound means
     something only where $\sigma_u \ll u$.

4. **The network.**
   - Use an MLP with input $n_k/N$ (24 numbers) and three outputs: $\hat\Omega$, $\hat u$, and
     a **signed** $\hat\Delta$. Use an MSE loss on all three, and draw fresh batches from the
     prior at every step, so the training set is unlimited and overfitting is not an
     issue.
   - Our reference run used two hidden layers of 128 tanh units, Adam at $10^{-3}$ with a
     cosine schedule, 20000 steps of batch 512. It trained in about 40 s on a laptop CPU.
     Any comparable choice is fine.
   - **The signed head.** Predict this before you train: the MSE-optimal output is the
     posterior mean, and $\pm\Delta$ are equally likely given the data. So $\hat\Delta$ should
     sit near $0$ for every input. Ours gave $|\hat\Delta| < 0.004$ at every test point. Print a
     CHECK on it.
   - **At (1.5, 0.5),** ours gave $\mathrm{std}(\hat\Omega) = 0.0127$, which is $1.14\times$ the
     CRB, and $\mathrm{rmse}(\hat u) = 0.0316$, which is $1.02\times$ the CRB. The network lands
     close to the bound but not on it, and it is unbiased in the interior of the prior.

5. **Where the network is good, and where it only looks good.**
   - Map the efficiency $\mathrm{CRB}^2/\mathrm{MSE}$ for $\hat\Omega$ and $\hat u$ over a grid of
     true parameters covering the prior, with a few hundred datasets per grid point. Show
     the two maps as heatmaps. Also split the MSE into bias$^2$ and variance, and show the
     bias map. The split is the point of this step.
   - What to expect. In the interior the efficiency is close to 1 and the bias is small. At
     the edges of the prior, the network pulls its estimates inward. Ours at
     $(1.02, 0.5)$ gave $\langle\hat\Omega\rangle = 1.028$. At $(2.0, 0.5)$ it gave
     $\langle\hat u\rangle = 0.267$ against the true $0.25$.
   - **The second apparent violation.** Wherever the variance alone dips below the CRB, find
     the bias that pays for it, and print both numbers. A network trained with MSE is
     trying to be the Bayes posterior mean, not an unbiased estimator. Near the edges of
     the prior it trades bias for variance, and the CRB has nothing to say about that
     trade.
   - **The small-$|\Delta|$ floor.** The reference network's $\hat\Omega$ error is flat at about
     $0.011$ for $|\Delta| \le 0.2$, while the bound falls to $0.0065$, so its efficiency drops to
     about $0.3$ there. A network with 3 hidden layers of 256, trained for 60000 steps of
     batch 1024 (about 8 min, so run it offline), brings this down to $0.0078$ at $\Delta = 0$
     and reaches the bound for $\Delta \ge 0.2$. The gap is capacity and training, and the CRB is
     what makes it visible.
   - Close this step by saying what the network actually learned. In the interior it is an
     efficient estimator that runs in microseconds, with no grid search and no
     multimodal likelihood. At the edges it reports the prior as much as the data.

## Push Harder

**The right yardstick for a Bayesian estimator.** On the same parameter grid as the MLE,
compute the exact posterior for each dataset. The likelihood is the binomial product, and
the prior is uniform. The posterior mean is the optimal MSE estimator for this prior, so it
is the true ceiling for the network, everywhere, including at the edges.

- Compare the network's MSE to the posterior mean's at the points from step 5. The gap is
  how much the network has left to learn.
- Then lower $N$ to 10 and 30, retrain, and repeat. Predict first. At small $N$ the prior
  matters more, the posterior mean should beat the CRB by a wide margin, and the MLE should
  fall off the bound because the asymptotic theory behind it no longer holds. Measure them
  and say whether the prediction held.
- What we measured. Averaged over the prior, the posterior mean beat the network and the
  MLE at every $N$, as it must. At $(1.5, 0.5)$, though, the second prediction failed: at
  $N = 10$ the posterior mean was $1.02$–$1.04\times$ the CRB and the MLE $1.04$–$1.06\times$.
  $24 \times 10 = 240$ shots already pin $\Omega$ far inside the prior. Also unpredicted: the
  network falls further behind as $N$ grows, from $1.03\times$ the CRB at $N = 10$ to $1.30\times$ at
  $N = 300$ in $\Omega$, which is the Step 5 floor again.

Plot RMSE against $N$ on log-log axes for the CRB, the MLE, the posterior mean and the
network at $(1.5, 0.5)$, with the $N^{-1/2}$ line. This section goes at the end of the
notebook.

## Out of Scope

- Decoherence, Ramsey sequences and $T_2^*$.
- Adaptive choice of measurement times (optimal experimental design).
- The quantum Fisher information and the Heisenberg limit.
- Networks that output their own error bars (a Gaussian-NLL head). It is a good idea, but
  it is a different session.

## Deliverables and Constraints

- **Layout.** Follow the previous sessions' layout:
  - The forward model, derivatives, Fisher matrix and CRB go in one `.py` module. The MLE
    (grid plus Fisher scoring) goes in another or the same one. The network and its
    training loop go in a third.
  - Fast checks go in a pytest `test_rabi.py` that runs well under a minute:
    - the analytic gradient against finite differences;
    - the CRB reference values from step 1 to four digits;
    - that the Fisher information for $\Delta$ is zero at $\Delta = 0$ while the one for $u$ is
      not;
    - that Fisher scoring agrees with `scipy.optimize.minimize` on a few datasets.
  - `build_nb.py` writes and executes `Rabi_Cramer_Rao.ipynb`.
  - The 2000-dataset MLE run, the network training and the efficiency maps run in the
    notebook. The printed `CHECK ... PASS/FAIL` lines live there too.
  - A `.gitignore` in the folder tracks only the `.md` and the `.ipynb`, as in previous
    weeks.
- **Notebook order.** The notebook follows the order of the steps above, with the headline
  figure first.
- **Libraries.** Use numpy and matplotlib for the model, the bound and the MLE. Use PyTorch
  for the network only. scipy is allowed in the test file only, as a cross-check of Fisher
  scoring.
- **Seeds.** Fix all seeds. Each statistical CHECK compares against its Monte Carlo error
  bar, not against a hard-coded tolerance.
- **Run time.** The whole notebook should run in about five minutes on a laptop. The
  efficiency maps dominate; vectorize the MLE across datasets rather than looping in
  Python.
- **Build incrementally.** Grow `build_nb.py` one section at a time, and rebuild and
  execute the notebook as each section lands. The notebook on disk should always show
  every result finished so far. A summary-of-checks cell always comes last.
- **Session length.** The session is meant to fit in about 100 minutes.
