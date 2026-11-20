# Snapshots of a Quantum Phase Transition

## The Problem

The manual day ran PCA on data whose structure was planted: a stretched Gaussian, and a
message hidden in ten dimensions. This session runs it on the kind of data a quantum
simulator produces. Here the structure is a phase transition, and nobody tells PCA what
the order parameter is.

The transverse-field Ising chain, with $J = 1$ and periodic boundaries,

$$H = -J\sum_{i=1}^{L} Z_i Z_{i+1} - h\sum_{i=1}^{L} X_i ,$$

has two simple limits. At $h = 0$ the ground states are all up and all down. As
$h \to \infty$ every spin points along $+x$. In between, at $h_c = J$ in the thermodynamic
limit, there is a quantum phase transition in the universality class of the 2D classical
Ising model ($\beta = 1/8$, $\nu = 1$).

A Rydberg-atom array or an ion trap does not give you $\psi$. Each run projectively
measures every spin in the $z$ basis and returns one bit string $s \in \{\pm 1\}^L$, drawn
with probability $|\psi(s)|^2$. That is a *snapshot*, and an experiment collects thousands
of them at each $h$. The question for the session is whether PCA can find the transition
from snapshots alone, and how much of what it finds is a discovery.

The classical version of this, PCA on Monte Carlo configurations of the 2D Ising model
from week 7, is L. Wang, Phys. Rev. B 94, 195105 (2016).

## Goal

Compute exact ground states, sample snapshots from them, and run PCA on the snapshots.
Show that the first principal component is the magnetization and that its
explained-variance ratio tracks $\langle m^2\rangle$, then locate $h_c$ by finite-size
scaling. Then say exactly what PCA computed.

The emphasis is on the exact counterpart. Every number PCA produces in this session has an
exact value that can be computed from the wavefunction, and every PCA result gets compared
with it.

## The Setup

- $J = 1$, periodic boundaries, $L \in \{6, 8, 10, 12, 14\}$. $2^{14} = 16384$ states is
  easy for a sparse solver.
- **Basis.** Label basis states by an integer $b$. Bit $i$ of $b$ gives
  $s_i = 1 - 2\,\mathrm{bit}_i$. $Z_i Z_{i+1}$ is diagonal, and $X_i$ flips bit $i$,
  $b \to b \oplus 2^i$. Build $H$ as a `scipy.sparse` matrix. Loop over the $L$ sites at
  most, never over the $2^L$ states. Get the ground state from
  `eigsh(H, k=1, which='SA')`.
- **Snapshots.** Draw `rng.choice(2**L, size=M, p=psi**2)`, then convert the indices to an
  $(M, L)$ array of $\pm 1$. The default is $M = 5000$.
- **Field grid.** Use $h$ from 0.1 to 2.0. Do not go below 0.1 without the parity fix
  described below.

**The exact benchmark.** A Jordan–Wigner transformation maps the chain to free fermions.
The ground state lies in the even sector of the parity $P = \prod_i X_i$, where

$$E_0 = -\sum_{n=0}^{L-1}\sqrt{1 + h^2 - 2h\cos k_n}, \qquad k_n = \frac{(2n+1)\pi}{L}.$$

At $h = 1$ this reduces to $E_0 = -2/\sin(\pi/2L)$. Two reference values are
$E_0 = -15.3225951511$ at $(L, h) = (12, 1)$ and $E_0 = -8.5090822351$ at $(8, 0.5)$. Our
ED matched the formula to $10^{-14}$. This is the first CHECK in the notebook.

**The parity trap.** For $h < 1$ the lowest even and odd states are split by a gap that
closes exponentially in $L$. Once that gap drops below the solver's tolerance, `eigsh`
returns an arbitrary mix of the two states, which is a symmetry-broken state. At $L = 12$
ours gave $\langle P\rangle = 0.99999$ at $h = 0.1$. At $h = 0.05$ it gave
$\langle P\rangle = 0.335$ and $\langle m\rangle = +0.94$.

- $P$ flips every bit, so in this basis $P\psi$ is `psi[::-1]`.
- Project the state, $\psi \leftarrow (\psi + P\psi)/\lVert\psi + P\psi\rVert$.
- CHECK that $\langle P\rangle = 1$ and that $E_0$ still matches the free-fermion value.

This is spontaneous symmetry breaking, getting in through the solver tolerance. The exact
ground state of a finite chain is a cat state, and its snapshots show both signs of $m$.

**Reference values.** The numbers in this brief come from a scratch smoke test run the
night before the session. They are there to check against, not to copy.

## Natural Steps

1. **Exact diagonalization and the benchmark.**
   - Build $H$, find $\psi_0$, and apply the parity projection.
   - CHECK $E_0$ against the free-fermion formula to $10^{-10}$ at several $(L, h)$,
     including $h = 1$.
   - CHECK $\langle P\rangle = 1$ at $L = 12$, $h = 0.05$, the case where the unprojected
     solver fails.
   - From $|\psi|^2$, compute the exact $\langle m^2\rangle$ and $\langle m^4\rangle$, with
     $m = \frac{1}{L}\sum_i s_i$, and the correlator $G(r) = \langle Z_0 Z_r\rangle$. These
     are the exact counterparts for every later step.

2. **Headline figure: what the experiment sees.**
   - Top row: 40 snapshots each at $h = 0.3$, $1.0$ and $2.0$ for $L = 12$, shown as
     black-and-white images with shots as rows and sites as columns. Expect rows of a single
     color at $h = 0.3$, with both colors appearing because the state is a cat. At $h = 1$
     expect domains, and at $h = 2$ salt-and-pepper.
   - Bottom: the explained-variance ratio of PC1 against $h$ for every $L$, with the exact
     $\langle m^2\rangle$ drawn as lines. Fill this in when step 3 lands.

3. **PCA on the snapshots.**
   - Center the snapshots, form the covariance and diagonalize it with `np.linalg.eigh`,
     exactly as on the manual day. Report EVR₁, the overlap of PC1 with the uniform vector
     $u = (1, \dots, 1)/\sqrt{L}$, and a histogram of the PC1 projections.
   - **Predict before you run it.** With $\langle Z_i\rangle = 0$ and $Z_i^2 = 1$,
     $\mathrm{Tr}\,C = L$. If PC1 is $u$, its eigenvalue is $u^T C u = L\langle m^2\rangle$.
     So EVR₁ $= \langle m^2\rangle$. It goes to 1 as $h \to 0$, and to $1/L$ as
     $h \to \infty$, where $C \to I$.
   - Reference values at $L = 12$, $M = 5000$:

     | $h$ | exact $\langle m^2\rangle$ | PCA EVR₁ | $\lvert \mathrm{PC1}\cdot u\rvert$ |
     |---|---|---|---|
     | 0.3 | 0.9787 | 0.9800 | 1.0000 |
     | 0.8 | 0.7936 | 0.7923 | 1.0000 |
     | 1.0 | 0.5601 | 0.5592 | 0.9999 |
     | 1.2 | 0.3405 | 0.3353 | 0.9999 |
     | 2.0 | 0.1552 | 0.1567 | 0.9916 |

   - The CHECK compares EVR₁ with $\langle m^2\rangle$ against the Monte Carlo error of a
     mean of $M$ samples of $m^2$, which is $\mathrm{std}(m^2)/\sqrt{M}$. Do not use a fixed
     tolerance.
   - The overlap drops at $h = 2$ because the top eigenvalues are close together there
     (1.88, 1.52, 1.47), and 5000 shots cannot resolve which one comes first. Say so in
     the notebook.
   - Deep in the ordered phase the histogram of PC1 projections has two peaks, and deep in
     the disordered phase it has one. Show where it changes for each $L$.

4. **What PCA actually computed.**
   - The state is translation invariant, so the covariance $C_{ij} = G(i - j)$ is a
     circulant matrix. Its eigenvectors are Fourier modes, and its eigenvalues are the
     structure factor $S(k) = \sum_r e^{ikr} G(r)$ with $k = 2\pi n/L$. PC1 is the $k = 0$
     mode because $S(k)$ peaks at $k = 0$ in a ferromagnet. So "PCA found the order
     parameter" means "$S(0)$ dominates the spectrum."
   - CHECK the PCA spectrum against the exact $S(k)$. Ours at $L = 12$, $h = 1$, from 20000
     snapshots:

     | | eigenvalues, largest first |
     |---|---|
     | exact $S(k)$ | 6.721, 0.908, 0.908, 0.536, 0.536, 0.401, 0.401, 0.338, 0.338, 0.307, 0.307, 0.298 |
     | PCA | 6.715, 0.925, 0.899, 0.545, 0.534, 0.404, 0.397, 0.338, 0.334, 0.307, 0.304, 0.299 |

     The eigenvalues come in pairs because $\pm k$ are degenerate. Also note that
     $S(0) = L\langle m^2\rangle$, which is step 3 again.
   - Close the step with the limitation: PCA sees only the two-point function. Anything in
     the snapshots beyond $\langle Z_i Z_j\rangle$ is invisible to it.

5. **Locating $h_c$.**
   - Use the Binder cumulant $U_L = 1 - \langle m^4\rangle / (3\langle m^2\rangle^2)$, where
     $m$ is the PC1 projection divided by $\sqrt{L}$. Once PC1 $= u$, this is the same as
     the snapshot mean. The curves for different $L$ cross near $h_c$.
   - **Exact first.** These crossings use $|\psi|^2$ on 81 points in $h \in [0.8, 1.2]$,
     with consecutive sizes crossed by linear interpolation:

     | sizes | crossing |
     |---|---|
     | 6, 8 | 0.9284 |
     | 8, 10 | 0.9572 |
     | 10, 12 | 0.9715 |
     | 12, 14 | 0.9797 |

     The shortfall from 1 falls as $\bar{L}^{-2}$, where $\bar{L}$ is the mean of the pair.
     Fitting $h^*(\bar{L}) = h_c + a/\bar{L}^2$ gives $h_c = 1.0008$, or $1.0004$ from the
     last three pairs alone. The whole exact sweep took 7 s.
   - **Then from snapshots only,** as an experiment would do it. We hit two traps:
     - Taking the first sign change of $U_a - U_b$ on noisy curves is biased low, because
       noise creates spurious early crossings. Ours came out about $2\sigma$ low. Fit a
       low-order polynomial to $U_a - U_b$ and take its root instead.
     - The fit window itself shifts the root. A quadratic on $[0.85, 1.15]$ with 31 points
       puts the *exact* crossings at 0.9521 and 0.9653 rather than 0.9572 and 0.9715.
       Compare the snapshot crossings with the exact curves processed the same way.
   - Reference values, using a quadratic fit on $[0.85, 1.15]$ with bootstrap errors from
     200 resamples:

     | snapshots per $h$ | sizes 8, 10 | sizes 10, 12 |
     |---|---|---|
     | 2000 | 0.973 ± 0.019 | 0.968 ± 0.016 |
     | 10000 | 0.964 ± 0.008 | 0.966 ± 0.008 |
     | 50000 | 0.949 ± 0.004 | 0.968 ± 0.004 |
     | exact, same fit | 0.9521 | 0.9653 |

   - Close the step with this point. At $10^4$ shots the statistical error is under 1%,
     while the finite-size shortfall at $L \le 12$ is 3%. The bottleneck is system size,
     not the number of shots.

## Push Harder

Predict first for each item, then measure.

- **The antiferromagnet.** Set $J = -1$ and work out what PC1 should be. Ours at
  $L = 12$, $h = 0.3$ was the staggered vector $(+, -, +, -, \dots)/\sqrt{12}$, with its
  Fourier peak at $k = \pi$ and a top eigenvalue of 11.73 out of 12. PCA finds the staggered
  magnetization without being told. The numbers match the ferromagnet's, because flipping
  every other spin maps one model onto the other. Use even $L$.
- **A frustrated chain (ANNNI).** Add $+\kappa\sum_i Z_i Z_{i+2}$, which penalizes aligned
  next-nearest neighbors. At $\kappa = 1$, $h = 0.3$, $L = 12$, ours gave two nearly equal
  top eigenvalues (5.92 and 5.83, then 0.03). PC1 had period 4,
  $\uparrow\uparrow\downarrow\downarrow$, at $k = \pi/2$. The order parameter has two
  components, the cosine and the sine at $k = \pi/2$, and PCA returns both. Use $L$
  divisible by 4. Then scan $(\kappa, h)$ and map which $k$ wins. The ANNNI phase diagram
  between these phases is known to be complicated, so treat the map as exploratory and do
  not over-read $L = 12$.
- **Readout error.** Real detectors misread each atom with a probability $p$ of a few
  percent. Flip each bit with probability $p$, and predict the effect first. The
  off-diagonal $C_{ij}$ should scale by $(1 - 2p)^2$ while the diagonal stays 1. Then the
  eigenvectors do not change, and $\lambda \to (1 - 2p)^2\lambda + 1 - (1 - 2p)^2$.
  - What we measured at $p = 0.03$, $L = 12$, $h = 0.8$, $M = 20000$: the mean off-diagonal
    $C$ went from 0.7747 to 0.6852. The ratio, 0.8845, compares with 0.8836 predicted.
    EVR₁ went from 0.7935 to 0.7115.
  - Invert the relation to correct EVR₁, and find out what uncorrected readout error does
    to the Binder crossing.
- **The wrong basis.** Measure in $x$ instead by applying a Hadamard to every site,
  $\psi_x = H^{\otimes L}\psi$. That is a reshape to $[2]^L$ and one `tensordot` per axis.
  Predict what PCA finds.
  - Ours, from the exact $x$-basis covariance at $L = 12$: EVR₁ stayed between 0.085 and
    0.16 at every $h$ we tried in $[0.3, 2]$, so there is no dominant component.
  - The transition shows up in $\langle X\rangle$, the mean that centering throws away.
    CHECK $\langle X\rangle$ from the snapshots against Hellmann–Feynman,
    $\langle X\rangle = -\frac{1}{L}\,\partial E_0/\partial h$. Both give 0.6384 at
    $L = 12$, $h = 1$.
  - The lesson is that what PCA can find depends on the measurement basis, and the
    experimenter chooses the basis.

## Out of Scope

- Dynamics: quenches, Kibble–Zurek, time evolution of snapshots.
- Finite temperature.
- Larger systems via DMRG or tensor networks.
- Entanglement entropy and the central charge $c = 1/2$. It is a good idea, but it is a
  different session.
- Nonlinear methods: autoencoders, diffusion maps, persistent homology on snapshots. Stay
  with PCA, so that every number has an exact counterpart.

## Deliverables and Constraints

- **Layout.** Follow the previous sessions' layout:
  - `tfim.py` holds the sparse Hamiltonian, the parity-projected ground state, the
    free-fermion energy, the snapshot sampler, and the exact moments and correlator from
    $|\psi|^2$.
  - PCA stays in plain numpy with `eigh`, a few lines long, as on the manual day. It can go
    in `tfim.py` or in a small second module.
  - Fast checks go in a pytest `test_tfim.py` that runs well under a minute:
    - $E_0$ against free fermions at several $(L, h)$, to $10^{-10}$;
    - $\langle P\rangle = 1$ after projection at $L = 12$, $h = 0.05$;
    - the PCA code, fed the exact covariance, returns $S(k)$ to machine precision;
    - EVR₁ from sklearn's `PCA` matches ours on one snapshot set.
  - `build_nb.py` writes and executes `Quantum_Ising_Snapshots.ipynb`. The printed
    `CHECK ... PASS/FAIL` lines live in the notebook.
  - A `.gitignore` in the folder tracks only the `.md` and the `.ipynb`, as in previous
    weeks.
- **Notebook order.** The notebook follows the order of the steps above, with the headline
  figure first.
- **Libraries.** Use numpy, `scipy.sparse` with `eigsh`, and matplotlib. sklearn is used
  in the test file only, as a cross-check.
- **Seeds.** Fix all seeds. Each statistical CHECK compares against its Monte Carlo or
  bootstrap error bar, not against a hard-coded tolerance.
- **Run time.** The whole notebook should run in about five minutes on a laptop. The
  snapshot bootstrap in step 5 dominates, so vectorize over resamples where you can.
- **Build incrementally.** Grow `build_nb.py` one section at a time, and rebuild and
  execute the notebook as each section lands. The notebook on disk should always show every
  result finished so far. A summary-of-checks cell always comes last.
- **Session length.** The session is meant to fit in about 100 minutes. Steps 1–4 are the
  core. Step 5 and Push Harder are where the remaining time goes.
