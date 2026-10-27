# Diffusion Monte Carlo

## The Problem

Last week's path integral put a quantum particle at finite temperature and sampled
$e^{-\beta H}$. Push $\beta \to \infty$ and only the ground state survives. Diffusion
Monte Carlo makes that limit into a process you can run. With $\hbar = m = 1$ the
Schrödinger equation in imaginary time $\tau = it$ is

$$-\frac{\partial \psi}{\partial \tau} = \left[-\tfrac12 \nabla^2 + V(\mathbf x) - E_T\right] \psi .$$

Expand any starting $\psi$ in eigenstates and every component decays like
$e^{-(E_n - E_T)\tau}$. The ground state decays slowest, so at long times it is the only
one left. If $E_T = E_0$ it neither grows nor shrinks.

Read the two terms on the right as two things a crowd of particles can do. The
$\tfrac12\nabla^2$ term is diffusion: each walker takes a Gaussian step of variance
$\Delta\tau$ in each direction. The $V - E_T$ term is a birth and death rate: where the
potential is low, walkers multiply, and where it is high, they die. Run the crowd for a
long time and its density *is* $\psi_0$. Notice that this is $\psi_0$ itself, not
$|\psi_0|^2$. Nothing in the process squares anything, which is a real difference from
last week and from variational Monte Carlo, and a free check that you got it right.

The bookkeeping that makes it work is $E_T$. Too high and the population explodes, too
low and it dies out. Adjust it as you go so the walker count stays near a target. The
value it settles to is the ground-state energy.

## Goal

Build a plain diffusion Monte Carlo, with no importance sampling and no trial
wavefunction, and show it projects out the ground state of the harmonic oscillator and
of hydrogen. Then understand its time-step error well enough to remove it.

Leave importance sampling out on purpose. It is more accurate, but it changes what the
walkers sample to $\psi_T \psi_0$, and the $\psi_0$ versus $|\psi_0|^2$ check is the
point.

## Natural Steps

1. One step for a population of walkers in $V = x^2/2$:
   - diffuse each walker, $x \to x + \sqrt{\Delta\tau}\,\xi$;
   - give it the weight $w = e^{-(V(x) - E_T)\Delta\tau}$;
   - replace it with $\lfloor w + u \rfloor$ copies, $u$ uniform on $[0,1)$.

   Keep the whole population in one numpy array and branch with `np.repeat`, with no
   Python loop over walkers. Steer $E_T$ with feedback on $\ln(N / N_\text{target})$.
   How hard to steer is your call. Steer too hard and $E_T$ gets noisy, too gently and
   $N$ wanders.

2. The first figure in the notebook is the energy converging. Start the walkers
   somewhere deliberately wrong, for example uniform on $[-4, 4]$. Plot $E_T(\tau)$ and
   the walker average $\langle V \rangle(\tau)$ against $\tau$, with the exact
   $E_0 = 1/2$ as a dashed line. The reader should see both start off and settle onto
   the line. A small panel of $N(\tau)$ underneath shows the population control
   working.

   Why is the average of $V$ over walkers an energy? Integrate $H\psi_0 = E_0\psi_0$ over
   all space. The kinetic term is a total derivative and vanishes, which leaves
   $\int V\psi_0 = E_0\int\psi_0$. Averaging $V$ over a density proportional to $\psi_0$
   gives $E_0$.

3. The two estimators have exact answers at finite $\Delta\tau$, for the same reason as
   last week: everything is Gaussian. Measured after branching, the walker density is a
   Gaussian of variance $s = \tfrac12\left(\sqrt{4 + \Delta\tau^2} - \Delta\tau\right)$, so

   $$\langle V \rangle_{\Delta\tau} = \frac{s}{2} \approx \frac12 - \frac{\Delta\tau}{4} + \frac{\Delta\tau^2}{16},
   \qquad
   \langle E_T \rangle_{\Delta\tau} = \frac{\operatorname{asinh}(\Delta\tau/2)}{\Delta\tau} = \frac{\tilde\omega}{2} \approx \frac12 - \frac{\Delta\tau^2}{48}.$$

   The $\tilde\omega$ is last week's lattice frequency,
   $\cosh(\Delta\tau\,\tilde\omega) = 1 + \Delta\tau^2/2$. One DMC step is one time
   slice of the path integral. At $\Delta\tau = 0.1$ these give $0.475625$ and
   $0.499792$. Your Monte Carlo should hit them within error bars. (They were checked
   against a direct diagonalization of the one-step transfer kernel on a grid.)

   The error bars need the same care as last week. Successive steps are strongly
   correlated, so reuse the binning from `06_Statistics_and_Monte_Carlo/errors.py`.
   Throw away the projection transient before averaging. Population control also adds a
   small bias that shrinks like $1/N_\text{target}$. Pick $N_\text{target}$ large enough
   that this bias sits below your error bar. Copy `errors.py` into this folder rather
   than importing it across folders.

4. The walker histogram, at late times and collected over many steps, overlaid with two
   curves. One is $\psi_0$ normalized as a density, a standard normal with $\sigma = 1$.
   The other is $|\psi_0|^2$, a normal with $\sigma = 1/\sqrt2$. The histogram should sit
   on the first. At finite $\Delta\tau$ its variance is $s$ from step 3, slightly below 1.
   That is the time-step error showing up in the shape.

5. The time-step error. Run several $\Delta\tau$ values, for example $0.4$, $0.2$,
   $0.1$, $0.05$, and plot both estimators against $\Delta\tau$ with their exact curves
   from step 3. The error in $\langle V\rangle$ starts at first order in $\Delta\tau$. A
   straight line is not enough, though: the $\Delta\tau^2/16$ term pulls a linear fit
   over this grid to $0.4979$, which is many error bars off. So fit
   $a + b\,\Delta\tau + c\,\Delta\tau^2$ and extrapolate to $\Delta\tau \to 0$. You
   should get $1/2$ (the exact values give $0.49999$). $E_T$ has no first-order term, since
   the primitive step $e^{-\Delta\tau V}e^{-\Delta\tau T}$ has the same eigenvalues as
   the symmetric split $e^{-\Delta\tau V/2}e^{-\Delta\tau T}e^{-\Delta\tau V/2}$. So
   extrapolate it in $\Delta\tau^2$. Both extrapolations should agree with $1/2$. Keep the
   total imaginary time roughly fixed across $\Delta\tau$, so small steps do not leave
   the projection unfinished.

6. Hydrogen. Walkers live in 3D, $V = -1/r$, $E_0 = -1/2$, $\psi_0 \propto e^{-r}$.
   Show the energy convergence again. Then show the radial histogram against two curves:
   $P(r) = \tfrac12 r^2 e^{-r}$ for $\psi_0$, and $4 r^2 e^{-2r}$ for $|\psi_0|^2$. The
   two peak at $r = 2$ and $r = 1$, so they are easy to tell apart.

   The Coulomb singularity is the hard part. A walker that lands near $r = 0$ gets an
   enormous weight $e^{\Delta\tau/r}$, and with it a burst of copies. You will need a
   guard, such as capping the copies per walker. Without one, the expected weight
   $\int r^2 e^{\Delta\tau/r}\,dr$ diverges at $r = 0$, so a guard is required, not
   optional. Say what you chose. Check that it does not move the answer by varying the
   cap, since there is no uncapped run to compare with. Expect more noise and a larger time-step error than the
   oscillator. Getting $-1/2$ within a few error bars, after extrapolating in
   $\Delta\tau$ (the form is up to you, by looking at the data), is the bar. There is no
   high-precision target here.

## Push Harder

The first excited state of the oscillator is odd, with a node at $x = 0$. Kill every
walker that crosses the origin and start them all on one side. The crowd now projects
onto the lowest state with that node, and $E_T$ should settle at $E_1 = 3/2$. That is
fixed-node DMC. In 1D the node is known exactly, so the answer is exact as
$\Delta\tau \to 0$. At finite $\Delta\tau$, a walker can cross the node and come back
within one step without being caught. Killing it with the crossing probability
$e^{-2xx'/\Delta\tau}$ removes most of that error. In many-fermion problems the node is
unknown. Guessing one costs a fixed-node error. That is the price of sidestepping the
sign problem, where the signal-to-noise ratio collapses exponentially if no node is
imposed.

This session does the fixed-node run. It goes at the end of the notebook, after hydrogen.

- Show $E_T$ settling to $3/2$.
- Show the walker histogram against $\psi_1$ normalized as a density on the half-line,
  $x\,e^{-x^2/2}$ for $x > 0$.
- Print a check that $E_1 = 3/2$ within error bars. Get there either with the crossing
  correction or by extrapolating in $\Delta\tau$, your choice.

Out of scope this time: importance sampling with $\psi_T = e^{-\alpha x^2/2}$. It would
move the histogram to $\psi_T\psi_0$ and cut the energy noise. That is exactly the trade
the main steps give up.

## Deliverables and Constraints

- Follow last week's layout. The algorithm goes in `.py` modules. Fast checks go in a
  pytest `test_dmc.py`, well under a minute: the step 3 exact values at one
  $\Delta\tau$, and the histogram width. `build_nb.py` writes and executes
  `Diffusion_Monte_Carlo.ipynb`. The long runs live in the notebook, and so do the
  printed `CHECK ... PASS/FAIL` lines: the extrapolations to $1/2$, and hydrogen within
  error bars. Only the `.md` and `.ipynb` are pushed, as last week.
- The notebook follows the order of the steps above. The energy convergence plot comes
  first.
- Use numpy and matplotlib only. Everything should run in a few minutes on a laptop.
  The session is meant to fit in about 100 minutes.
