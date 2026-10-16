# Path Integral Monte Carlo

## The Problem

Metropolis samples a classical Boltzmann weight $e^{-E/T}$. Quantum mechanics has no
such weight, it has operators. But Feynman's path integral turns one into the other.
Write the thermal partition function of a particle in a potential $V(x)$ as a trace,
split the imaginary time $\beta$ into $M$ slices of width $\varepsilon = \beta/M$, and
insert a complete set of position states between each pair. With $\hbar = m = 1$,

$$Z = \mathrm{Tr}\, e^{-\beta H} \approx \int dx_0 \cdots dx_{M-1}\; e^{-S_E},
\qquad S_E = \sum_{j=0}^{M-1} \left[ \frac{(x_{j+1}-x_j)^2}{2\varepsilon} + \varepsilon\, V(x_j) \right],
\qquad x_M \equiv x_0 .$$

Look at what that is. One quantum particle in one dimension has become a *classical*
ring of $M$ beads joined by springs of stiffness $1/\varepsilon$, each bead sitting in
the potential $\varepsilon V$. The ring closes because of the trace. The extra
dimension is imaginary time, and $e^{-S_E}$ is an ordinary Boltzmann weight you
already know how to sample. A $d$-dimensional quantum system is a $(d+1)$-dimensional
classical one, which is the idea behind lattice QCD.

The harmonic oscillator, $V = x^2/2$, is the place to start, since everything about it
is known:

$$\langle x^2 \rangle = \tfrac12 \coth(\beta/2),$$

which is the zero-point value $1/2$ at low temperature and equipartition, $1/\beta$, at
high temperature. One simulation, swept in $\beta$, has to get both.

## Goal

Build a path integral Monte Carlo for the oscillator and get the right answer, with an
error bar you can defend. Then point it at a potential with no closed-form answer.

## Natural Steps

1. Metropolis on the ring: move one bead at a time and accept on the change in $S_E$.
   Only the two springs touching the bead and its own potential term change, so the
   update is local and cheap.
2. Measure $\langle x^2 \rangle$ against $\beta$ and compare with $\tfrac12\coth(\beta/2)$.
3. Separate the two kinds of error. $S_E$ is quadratic for the oscillator, so the
   discretized problem is exactly solvable at finite $M$:
   $$\langle x^2 \rangle_M = \frac{\varepsilon}{2\sinh(\varepsilon\tilde\omega)}\coth\!\left(\frac{\beta\tilde\omega}{2}\right),
   \qquad \cosh(\varepsilon\tilde\omega) = 1 + \frac{\varepsilon^2}{2}.$$
   Your Monte Carlo should hit $\langle x^2\rangle_M$ within its error bars, and
   $\langle x^2 \rangle_M$ should approach the continuum answer like $\varepsilon^2$.
   (At $\beta = 10$, $M = 20$ this is $0.48512$, against a continuum $0.50005$.)
4. The beads are correlated along the chain and from one sweep to the next, so the
   naive $\sigma/\sqrt N$ from the manual session is too small. Measure the
   autocorrelation time and fix the error bar, by binning or otherwise.
5. At low temperature the histogram of all the bead positions is the ground-state
   density, $|\psi_0(x)|^2 = \pi^{-1/2} e^{-x^2}$. Plot it.

## Push Harder

The spectrum is in there too. The imaginary-time correlator
$\langle x(0)\, x(\tau)\rangle$ decays like $e^{-(E_1 - E_0)\tau}$, so fitting its
slope measures the gap, which is $1$ for the oscillator (on the lattice it is
$\tilde\omega$ above, slightly less). This is how lattice QCD gets the proton mass.
Think about which $\tau$ to fit, because the signal drops into the noise fast.

As $M$ grows the springs get stiffer and single-bead moves stop getting anywhere. Watch
the autocorrelation time grow with $M$, then do something about it: a move that shifts
the whole ring rigidly, or updates in the Fourier modes of the ring, which for the
oscillator are independent.

Then the real payoff: a double well, $V = (x^2 - a^2)^2$, where there is no formula.
Your finite-difference eigensolver from the quantum session can give you $E_0$, $E_1$
and $\langle x^2 \rangle$ exactly, so you still have a check. Make the barrier tall
and look at individual paths. They sit in one well and occasionally jump to the
other. Those jumps are instantons, and they are the tunneling, seen directly.
