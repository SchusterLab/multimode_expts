# Gain derivation: from drive amplitude to QICK register

This document derives, precisely, the relationship between an envelope's peak
drive amplitude and the QICK gain register that produces it, for the qubit
drive on the v1 tProc used here. It pins down the **Hamiltonian and gauss
conventions** that `compute_gains_from_ghz` (and its source,
[`MM_base.get_gain_optimal_pulse`](../../experiments/MM_base.py#L2132)) silently
assume, so that Harmoniqs-supplied optimal-control pulses are mapped to gain
registers without silent unit errors.

The earlier version of this note claimed a "factor-of-2 bug" in the formula.
That claim was wrong — it assumed the standard RWA convention
$H = (\Omega_R/2)\,\sigma_x$. The codebase consistently uses the convention
$H = \Omega\,\sigma_x$ (no $1/2$) for the drive, which makes the formula
internally consistent. See §5 below for the resolution.

## 1. The played envelope (`GAUSS_BUG` is live on v1)

QICK's helper-level gauss is the standard normal form
([`qick/helpers.py:95`](../../../../python/qick/qick_lib/qick/helpers.py#L95)):

$$g_{\text{helper}}(t) = G\,\exp\!\left(-\frac{(t-\mu)^2}{2\sigma_{\text{helper}}^2}\right)$$

But the program-level `add_gauss` method
([`qick/qick_asm.py:1369–1370`](../../../../python/qick/qick_lib/qick/qick_asm.py#L1369))
has a flag `GAUSS_BUG`, set `True` for the v1 tProc in
[`asm_v1.py:564`](../../../../python/qick/qick_lib/qick/asm_v1.py#L564), that
divides the user-supplied `sigma` by $\sqrt{2}$ before calling the helper:

```python
class QickProgram(...):
    GAUSS_BUG = True  # "incorrect original definition, narrower by sqrt(2)"

def add_gauss(self, ..., sigma, length, ...):
    if self.GAUSS_BUG:
        sigma /= np.sqrt(2.0)
    self.add_envelope(ch, name, idata=gauss(..., si=sigreg, length=lenreg, ...))
```

We're on v1 (firmware reports `axis_tproc64x32_x8 ("v1") rev 4`), so the bug
is live. [`MM_base.py:333`](../../experiments/MM_base.py#L333) creates the π
envelope as

```python
self.add_gauss(..., sigma=self.pi_ge_sigma, length=self.pi_ge_sigma*4)
```

with `pi_ge.sigma` the un-divided config value ($\sigma_\pi = 0.035\ \mu s$).
The played envelope is therefore

$$\boxed{\;g_{\text{played}}(t) = G\,\exp\!\left(-\frac{(t-2\sigma)^2}{\sigma^2}\right),\quad t\in[0,\,4\sigma],\ \sigma\equiv\sigma_\pi\;}$$

i.e. the `exp(-x^2/sigma^2)` form (no $/2$ in the exponent), peak at centre,
truncated at $\pm 2\sigma$ around the peak.

## 2. Integrated envelope area (unit peak)

$$\mathcal{A}(\sigma) \;\equiv\; \int_{-2\sigma}^{+2\sigma}\!\exp\!\left(-\frac{u^2}{\sigma^2}\right)du = \sigma\int_{-2}^{+2}\!e^{-v^2}\,dv = \sigma\sqrt{\pi}\,\mathrm{erf}(2)$$

This is exactly the factor `sigma * sqrt(pi) * erf(n/2)` (with $n=4$) that
appears in `compute_gains_from_ghz`. Numerically:

$$\mathcal{A}(\sigma_\pi) = 0.035 \times 1.77245 \times 0.99532 = 0.06175\ \mu s$$

## 3. Hamiltonian convention (the key point)

The convention this codebase consistently uses for the **qubit drive
Hamiltonian** is

$$H_{\text{drive},q}(t) \;=\; \Omega(t)\,(q+q^\dagger) \;=\; \Omega(t)\,\sigma_x \quad (\text{truncated to 2-level})$$

i.e. $\Omega$ is the **coefficient of $\sigma_x$**, with **no factor of $1/2$**.
This is *not* the standard RWA form $H=(\Omega_R/2)\sigma_x$ where $\Omega_R$
is the angular Rabi rate; it differs from it by a factor of 2 in the
parametrisation (same physics, different bookkeeping).

Under this convention, the rotation angle from a pulse is

$$\theta \;=\; 2\int \Omega(t)\,dt \;=\; 2\,\Omega^{\text{peak}}\,\mathcal{A}(\sigma)$$

(the factor of 2 comes from $H = \Omega\sigma_x$ generating $U = e^{-i\theta\sigma_x/2}$
with $\theta = 2\int\Omega\,dt$ — equivalently the physical angular Rabi rate
is $\Omega_R = 2\Omega$).

**Harmoniqs ship $\Omega$ in the same convention** (per their `manifest.json`:
$H_{\text{drive},q} = \Omega_I\,\hat{X}_q + \Omega_Q\,\hat{P}_q$, with the note
"Rabi rate (rad/ns) = $2|\Omega|$"), so no factor-of-2 reconciliation is
needed between the two sides.

## 4. Anchoring `pi_ge.gain` to a true $\pi$ rotation

Empirical evidence from the pinned config `CFG-HW-20260527-00049`:

| pulse  | gain |
|--------|------|
| `pi_ge`  | 9993 |
| `hpi_ge` | 4901 |

`hpi_ge.gain` is within 2% of `pi_ge.gain / 2`. The naming (`hpi` = half-pi)
together with linear-amplitude scaling ($\theta \propto G$) fixes

$$\theta(\text{hpi\_ge.gain}) = \pi/2 \;\Longrightarrow\; \theta(\text{pi\_ge.gain}) = \pi$$

i.e. `pi_ge.gain = 9993` is the true π-rotation gain. Substituting into
$\theta = 2\Omega^{\text{peak}}\mathcal{A}$:

$$2\,\Omega^{\text{peak}}(\text{gain}_\pi)\,\mathcal{A}(\sigma_\pi) = \pi
\;\Longrightarrow\;
\Omega^{\text{peak}}(\text{gain}_\pi) = \frac{\pi}{2\,\mathcal{A}(\sigma_\pi)} = \frac{\pi/2}{0.06175\ \mu s} \approx 25.44\ \text{rad}/\mu s$$

That's the σ_x coefficient at peak — the quantity in $H = \Omega\sigma_x$.
The corresponding physical angular Rabi rate is twice that:

$$\Omega_R^{\text{peak}}(\text{gain}_\pi) = 2\Omega^{\text{peak}} \approx 50.87\ \text{rad}/\mu s$$

In linear frequency:

$$\boxed{\;f_R^{\text{peak}}(\text{pi\_ge.gain} = 9993) \;=\; \frac{\Omega_R^{\text{peak}}}{2\pi} \;\approx\; \mathbf{8.10\ \text{MHz}}\;}$$

## 5. The formula in `compute_gains_from_ghz`: now consistent

[`MM_base.py:2183`](../../experiments/MM_base.py#L2183) /
[`core.py:252`](core.py#L252):

```python
theta_to_gain   = np.pi/2 / gain_pi                         # "half-rotation"/gain
drive_to_gain   = sigma * sqrt(pi) / theta_to_gain * erf(2) # gain/(rad/μs)
gain            = round(max_q * drive_to_gain)
```

Rearranging:

$$\text{gain} \;=\; \text{max\_q}\,\cdot\,\frac{\mathcal{A}(\sigma)\cdot\text{gain}_\pi}{\pi/2}$$

At calibration ($\text{gain} = \text{gain}_\pi$), this enforces
$\text{max\_q} \cdot \mathcal{A}(\sigma) = \pi/2$, i.e.
$\text{max\_q}(\text{gain}_\pi) = (\pi/2)/\mathcal{A}(\sigma_\pi) \approx 25.44$ rad/μs.

Under the codebase's $H = \Omega\sigma_x$ convention this is exactly
$\Omega^{\text{peak}}(\text{gain}_\pi)$ from §4 — so **`max_q` in the formula
is the σ_x coefficient in rad/μs**, and the formula's `theta_to_gain`
correctly represents the *σ_x-coefficient-area-per-gain*, even though its
literal name reads as "rotation angle per gain" (off by $\times 2$).
That's just a misleading variable name; the math is right.

## 6. Cavity branch (no convention issue)

The cavity branch uses the displacement calibration directly:

```python
drive_to_gain_cav = sigma_cav * sqrt(pi) / gain_to_alpha * erf(n/2)
gain_cav          = round(max_c * drive_to_gain_cav)
```

calibrated against $\alpha = G\cdot\text{gain\_to\_alpha}$ for a gauss pulse
of area $\mathcal{A}(\sigma_{\text{cav}})$. No implicit $\pi/2$ convention here:

$$\alpha = \varepsilon^{\text{peak}}\,\mathcal{A}(\sigma_{\text{cav}}) = G\cdot\text{gain\_to\_alpha} \;\Longrightarrow\; G = \varepsilon^{\text{peak}}\cdot\text{drive\_to\_gain\_cav}$$

So `max_c` is the cavity drive coefficient $\varepsilon$ in rad/μs.
Harmoniqs ship $\varepsilon$ in rad/ns in the matching convention
($\dot\alpha = -i\varepsilon$), so the conversion is plain unit math:
**cavity factor $= 1/(2\pi)$**.

## 7. Conversion factors for Harmoniqs envelopes

Identifying Harmoniqs' σ_x coefficient $\Omega$ (rad/ns) with the formula's
`max_q` (rad/μs):

$$\text{max\_q}\,[\text{rad}/\mu s] \;=\; \Omega\,[\text{rad}/\text{ns}] \cdot 10^3$$

The formula consumes `data['I_q'][GHz]` with
`max_q = data['I_q'] * 2π * 1e3`, so

$$\text{data}['I_q']^{[\text{GHz}]} \;=\; \frac{\Omega}{2\pi}$$

— and **the qubit factor is $1/(2\pi)$**, symmetric with the cavity. The
earlier `1/\pi` choice was wrong (it followed from incorrectly identifying
`max_q` with the angular Rabi rate $\Omega_R = 2\Omega$).

In code:

```python
# job_server/closed_loop/pulse_io.py
QUBIT_RAD_PER_NS_TO_GHZ  = 1.0 / (2.0 * np.pi)   # I_q[GHz] = Omega / (2π)
CAVITY_RAD_PER_NS_TO_GHZ = 1.0 / (2.0 * np.pi)   # I_c[GHz] = epsilon / (2π)
```

## 8. Harmoniqs fock1 gains under the corrected conversion

Peak Harmoniqs $|\Omega|_{\text{peak}} = 0.04942$ rad/ns ⇒
$\text{max\_q} = 0.04942 \times 10^3 / (2\pi) \times 2\pi \times 10^3 = 49.42$ rad/μs
(σ_x coefficient).

$$\text{gain}_{qb}^{\text{Harmoniqs fock1}} \;=\; \text{max\_q}\cdot\frac{\mathcal{A}(\sigma_\pi)\cdot\text{gain}_\pi}{\pi/2} \;=\; 49.42 \cdot \frac{0.06175 \cdot 9993}{\pi/2} \;\approx\; \mathbf{19{,}400}$$

— **in range** (< 32,767). The earlier 38,729 came from using the wrong
qubit factor ($1/\pi$ instead of $1/(2\pi)$), which doubled the effective
σ_x coefficient.

## 9. Why the dry-run looked fine (and an empirical sanity check)

The dry-run computed gains against the **repo copy** of `hardware_config.yml`,
which has `pi_ge.gain = 3289`. The pinned config has `pi_ge.gain = 9993`,
about 3× bigger. Both copies feed the same formula, so the *Harmoniqs* gain
scales in lock-step with `pi_ge.gain`. With the corrected qubit factor and
the live `pi_ge.gain = 9993`, the gain is ~19,400 (in range).

A one-time empirical check is still cheap and worth doing: sweep gain on a
single-qubit Rabi experiment at `pi_ge.sigma`, fit the first π maximum.
If it lands at 9993 ± 5%, this derivation holds end-to-end. If it lands
elsewhere, the codebase's $H = \Omega\sigma_x$ assumption (or the
`hpi:pi = 1:2` calibration) has drifted and the conversion needs another
adjustment.
