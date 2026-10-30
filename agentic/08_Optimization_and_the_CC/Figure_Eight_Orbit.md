# The Figure-Eight Orbit, Found by Optimization

## The Problem

The manual day ran simulated annealing and a genetic algorithm on an integer objective.
There was no derivative to take, and the landscape had a huge number of local minima.
This session uses an objective that is smooth, has an exact gradient, and has a physical
answer that is known to eight digits. The question is how much the gradient buys you.

The objective is the classical action. Three equal masses ($G = m = 1$) move in a plane
with period $T$. Their action is

$$S[q] = \int_0^T \Big[\sum_i \tfrac12 |\dot q_i|^2 + \sum_{i<j} \frac{1}{|q_i - q_j|}\Big]\,dt .$$

Newton's equations are $\delta S = 0$. Every stationary point of $S$ among $T$-periodic
paths is a real periodic orbit. So you can find orbits without integrating anything:
write the path in a finite basis and minimize $S$ over the coefficients.

The target is the figure-eight choreography. Moore found it numerically in 1993, and
Chenciner and Montgomery proved it exists in 2000. The three bodies chase each other
around a single figure-eight curve, a third of a period apart. Simó's initial conditions
are

$$q_1 = -q_2 = (0.97000436,\,-0.24308753),\quad q_3 = 0,\qquad
\dot q_3 = -2\dot q_1 = -2\dot q_2 = (-0.93240737,\,-0.86473146),$$

with period $T = 6.32591398$. Use exactly this $T$ throughout, so your numbers compare to
the references below without rescaling.

## Goal

Find the figure-eight by minimizing the action from random starting curves. Then race
five optimizers on that same objective and explain the race: why some methods need tens
of evaluations and others need hundreds of thousands. Finally, drop the symmetry that
singles out the eight, and show that the landscape then has two basins and the eight is
not the lowest one.

The emphasis is on the optimizers. The physics supplies an objective with known answers.

## The Parametrization

Use a **choreography**: one curve $q(t)$, with $q_j(t) = q(t + jT/3)$ for $j = 0, 1, 2$. All
three pairs then contribute the same integral, so

$$S = 3\int_0^T \Big[\tfrac12|\dot q(t)|^2 + \frac{1}{|q(t) - q(t + T/3)|}\Big]\,dt .$$

Write $q(t)$ as a Fourier series in $\omega = 2\pi/T$. Leave out every $k$ divisible by 3,
because those modes are exactly what moves the center of mass. With them gone the center
of mass sits at the origin automatically.

**The eight subspace.** Fourier-analysing Simó's orbit, with the body that is at the
origin at $t = 0$, shows that only these terms survive:

$$x(t) = \sum_{k\ \text{odd},\ 3\nmid k} a_k \sin k\omega t, \qquad
y(t) = \sum_{k\ \text{even},\ 3\nmid k} b_k \sin k\omega t .$$

Use $k \le K = 45$. That gives 30 coefficients. The leading ones are
$a_1 = 1.10084$, $b_2 = 0.33881$ and $b_4 = 0.05596$, up to an overall sign on $x$ or on
$y$. Flipping either sign is a reflection of the same orbit.

**The full choreography space.** Allow both $\cos$ and $\sin$ for both $x$ and $y$, with
$3 \nmid k$. Use $K = 30$ here, which gives 80 coefficients. This space contains the eight,
and it also contains every rotation and time shift of the eight.

**Evaluating $S$.**
- The kinetic term is exact in Fourier space: each mode contributes
  $3\cdot\tfrac12 (k\omega)^2 c_k^2 \cdot T/2$.
- The potential term uses $N$ equally spaced time samples with $N$ divisible by 3, so that
  the $T/3$ shift is an `np.roll` by $N/3$. A plain Riemann sum over the period is
  spectrally accurate. $N = 12K$ is plenty.
- Write $q(t_n)$ as basis matrices times the coefficient vector, so that evaluating $S$
  involves no Python loops.

**The gradient, by hand.** No autograd is available. With $d_n = q(t_n) - q(t_n + T/3)$,
the potential gradient comes from $\partial(1/|d|)/\partial d = -d/|d|^3$. It is then
pushed back through the roll and through the basis matrices. Check it against central
finite differences; the relative error should be $\lesssim 10^{-6}$. That check is the first
CHECK line in the notebook.

## Natural Steps

1. **The action and its gradient.**
   - Write the action and its gradient for both bases, and run the finite-difference check
     on each.
   - The exact anchor is the Lagrange equilateral circle. With $R^3 = 1/(\sqrt3\,\omega^2)$,
     its action is $S_L = 3T\big[\tfrac12\omega^2R^2 + 1/(\sqrt3 R)\big] = 19.648667$.
     Plug its coefficients into your choreography-space action. You should get $S_L$ to
     about $10^{-10}$.

2. **Headline figure: the eight appearing.**
   - Start BFGS in the eight subspace from a random curve, with
     $c_k \sim \mathcal N(0, 1)/k^2$.
   - Plot the curve $q(t)$ at a handful of iterations: the start, a few intermediate
     iterations, and the converged result. The reader should watch a scribble turn into
     the eight.
   - On the final curve, mark the three bodies at $t = 0$.
   - This is the first figure in the notebook.

3. **Is it really the orbit? Two checks that do not use the optimizer.**
   - **(a) The action value.**
     - The converged value is $S^* = 24.42704864077$. That number is from $K = 45$, and
       raising $K$ further does not change it.
     - Integrating Simó's 8-digit initial conditions independently gives
       $S = 24.4270486$, $E = -1.2871420$ and angular momentum $L = 0$.
     - Show $S$ against $K$ for $K$ from 4 to 45. It converges spectrally: the error at
       $K = 4, 6, 10, 15, 20$ is roughly $8\times10^{-2}$, $1\times10^{-2}$,
       $1\times10^{-4}$, $8\times10^{-7}$ and $1\times10^{-9}$.
   - **(b) Closure under the equations of motion.**
     - First polish the coefficients with a few Newton steps. Use the Hessian obtained by
       finite-differencing the analytic gradient; step 5 needs that code anyway.
     - BFGS stalls around $|\nabla S| \sim 10^{-8}$, because Armijo can no longer resolve
       differences in $S$. Converging $S$ to $10^{-8}$ says little about the path, since
       the error in $S$ is second order in the error of the path.
     - Closure at the $10^{-10}$ level needs $|\nabla S| \sim 10^{-13}$. Newton gets
       there, to about $10^{-15}$, in two or three steps. This adds a sixth method to the
       story, with quadratic convergence that is visible in the printout.
     - Then take the bodies' positions and velocities at $t = 0$. The velocities come from
       the analytic derivative of the series.
     - Integrate one period with a hand-written RK4 using $5000$ steps. Check that the
       state returns to itself.
     - At $K = 45$ the closure error $|y(T) - y(0)|$ is about $10^{-10}$. At $K = 30$ it is
       about $5\times10^{-6}$, and at $K = 20$ about $2\times10^{-5}$.
     - Also check that the energy is constant along the integration and that $L = 0$.
     - Compare the initial conditions with Simó's, up to the sign flips. They should agree
       to about $10^{-7}$; his values only have 8 digits.
   - Make the point in the notebook that $S$ converges much faster than the path does.
     The action is stationary at the true orbit, so its error is second order in the
     error of the path. This is exactly why the action is a good thing to optimize.

4. **The race.** All five methods run in the eight subspace at $K = 45$, from the same
   random starts. Plot $S - S^*$ against the number of objective evaluations on log-log
   axes, one curve per method, with the median over a few seeds. Count one call of
   action-and-gradient as one evaluation for the gradient methods. For the gradient-free
   methods, count one call of the action alone. State in the notebook that the gradient
   roughly doubles the cost of a call, so the comparison is not distorted by orders of
   magnitude. The five methods:
   - **Random search.** Draw curves from the same $\mathcal N(0,1)/k^2$ prior and keep the
     best.
     - It lands in the right basin but stalls. The best-so-far is about $0.6$ after
       $10^3$ draws, $0.3$ after $10^4$ and $0.2$ after $10^5$.
     - Explain this with a volume argument. Near the minimum, $S - S^*$ is quadratic in
       $d_\text{eff}$ effective directions. The best of $n$ draws therefore improves only
       like $n^{-2/d_\text{eff}}$.
     - Fit the effective dimension from the measured curve. It is the same curse of
       dimensionality as the manual day's Problem 1, seen from a different angle.
   - **Simulated annealing.** Reuse the manual day's scheme: perturb one coefficient by a
     Gaussian scaled by $1/k^2$, then apply Metropolis acceptance and geometric cooling
     from $T = 1$ to $10^{-9}$.
     - Use a **fixed** step of about $0.05/k^2$. A step that shrinks like $\sqrt{T}$, as
       in the manual day's Thomson problem, freezes near $S - S^* \approx 0.3$.
     - Expect about $2\times10^{-6}$ at $1.5\times10^5$ evaluations, and about
       $5\times10^{-8}$ at $2\times10^5$.
   - **Plain gradient descent,** with a backtracking (Armijo) line search. Expect about
     $5000$–$7500$ evaluations. Make the prediction in the notebook first, then correct it:
     - **Predict.** The kinetic term's Hessian is diagonal with entries $\propto k^2$. That
       gives $\kappa \approx (k_\max/k_\min)^2 = 44^2 = 1936$.
     - **Measure.** The true Hessian at $S^*$ has eigenvalues from $21.8$ to $18135$, so
       $\kappa \approx 833$. The potential stiffens the low-$k$ modes: the smallest
       eigenvalue is $21.8$, against $9.37$ for the kinetic $k = 1$ entry. Print the
       predicted and measured $\kappa$ side by side.
     - The order of magnitude is right, which is what the prediction is for. Steepest
       descent's cost scales with $\kappa$.
   - **Preconditioned gradient descent.** Divide the gradient by the kinetic-Hessian
     diagonal, $3\cdot\tfrac12 T (k\omega)^2$.
     - This is one line of code. It brings the condition number down to about $3.8$, and
       the method needs about 35–45 evaluations.
     - The fix is the physics: you are stepping in the metric the kinetic energy defines.
       Print the preconditioned $\kappa$ as well.
   - **BFGS.** Write it yourself, about 30 lines: an inverse-Hessian update, plus the same
     backtracking line search starting from step $1$. Run two variants and plot both.
     - **$H_0 = I$.** BFGS has to discover the conditioning on its own. It takes about 40
       iterations but about $370$ evaluations, and most of those are backtracking
       halvings, because a unit step in the identity metric is wildly too long.
     - **Scaled $H_0$.** After the first step, reset $H_0 = (s\cdot y / y\cdot y)\,I$.
       This is the standard Nocedal–Wright one-line fix. It takes about 60–85
       evaluations.
     - The story to tell: preconditioned GD wins, because a good metric from physics
       beats a metric learned on the fly. BFGS learns the metric, but it pays for that in
       line-search evaluations. scipy's BFGS, used as the test-file cross-check, takes
       about 50–60.

   **Collisions.** $1/r$ diverges when two bodies meet. A line-search trial or an annealing
   move can land on a near-collision and return a huge or non-finite $S$. Reject such
   points explicitly. In the line search, a non-finite trial counts as a failed step: halve
   the step and try again. Do not wrap anything in `try/except`.

5. **The landscape without the symmetry.** Move to the 80-dimensional choreography space.
   - Run BFGS from 100 random starts. Each run should end at one of two values:
     $S_L = 19.6487$ (the Lagrange circle) or $S^* = 24.4270$ (the eight).
   - Report the fractions. Expect roughly three quarters Lagrange and one quarter eight
     (77/23 with seeds 0–99). Seeds that end anywhere else must be printed and looked at,
     not silently dropped.
   - Compute the Hessian at each minimum by finite-differencing your analytic gradient.
     - The eight has exactly **two** zero eigenvalues, from rotation and time shift. All
       the others are positive, so it is a genuine local minimum.
     - The Lagrange circle has exactly **one** zero eigenvalue, because for a circle a
       rotation *is* a time shift.
     - Counting zero modes this way is a check that the gradient and the basis are right.
   - The lesson for the race: in this space a better global optimizer would land on the
     Lagrange circle. The eight is selected by the constraint you impose, not by how hard
     you optimize.

## Push Harder

**A good global optimizer finds the wrong answer.** Run simulated annealing in the
80-dimensional choreography space of step 5. Use $2\times10^5$ evaluations, about 40 s per
run, and seeds 0–2. Run it at two settings.

- **Hot.** Use $T_0 = 10$ and a fixed step of $0.2/k^2$.
  - Expect all three seeds to end on the Lagrange circle, $S_L = 19.6487$. That is the
    global minimum, and it is not the orbit we were after.
- **Cold.** Use $T_0 = 1$ and a step of $0.05/k^2$. These are the eight-subspace settings
  from step 4.
  - Expect a mix of outcomes. Our check gave eight, circle, eight for seeds 0, 1, 2.
  - Annealing started too cold is a randomized local search. It splits between the basins
    much as multistart BFGS does.

Show both outcomes as a small panel of final curves, one per seed. Print a CHECK that the
hot run puts every seed on $S_L$ to $10^{-4}$.

Close the notebook with the punchline. Annealing becomes a genuine global method only
when the temperature can carry it over the barrier between basins, and once it is, it
reliably returns the circle. You get the eight by choosing the space you search, the
subspace from step 2, not by choosing a stronger optimizer.

This section goes at the end of the notebook, after step 5.

## Out of Scope

- Other choreographies from Simó's catalogue.
- Linear stability (Floquet) analysis of the eight.
- Unequal masses and three dimensions.
- The genetic algorithm. Simulated annealing already represents the gradient-free
  family, and a second one adds runtime without a new idea.

## Deliverables and Constraints

- **Layout.** Follow the previous sessions' layout:
  - The algorithm goes in `.py` modules: the action, gradient and bases in one, and the
    optimizers in another or the same one. Reference values, the Lagrange formula and the
    RK4 closure check go in `reference.py`.
  - Fast checks go in a pytest `test_figure_eight.py` that runs well under a minute:
    - the gradient finite-difference check on both bases;
    - the Lagrange value $S_L$;
    - BFGS reaching $S^*$ to $10^{-8}$ in the eight subspace;
    - the RK4 closure at $K = 45$ below $10^{-8}$, after the Newton polish.
  - `build_nb.py` writes and executes `Figure_Eight_Orbit.ipynb`.
  - The race, the multistart and the Hessians run in the notebook. The printed
    `CHECK ... PASS/FAIL` lines live there too.
  - A `.gitignore` in the folder tracks only the `.md` and the `.ipynb`, as in previous
    weeks.
- **Notebook order.** The notebook follows the order of the steps above, with the eight
  appearing first.
- **Libraries.** Use numpy and matplotlib only in the modules and the notebook. The
  optimizers and RK4 are hand-written; they are the point of the session. scipy is
  allowed in the test file only, as a cross-check of the hand-written BFGS against
  `scipy.optimize.minimize`.
- **Run time.** The whole notebook should run in about six minutes on a laptop.
  Simulated annealing dominates: about 15 s per $10^5$ evaluations in the eight subspace,
  and about 4 min for the six Push Harder runs.
- **Build incrementally.** Grow `build_nb.py` one section at a time, and rebuild and
  execute the notebook as each section lands. The notebook on disk should always show
  every result finished so far. A summary-of-checks cell always comes last.
- **Session length.** The session is meant to fit in about 100 minutes.
