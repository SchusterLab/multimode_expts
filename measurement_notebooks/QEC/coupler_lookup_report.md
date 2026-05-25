---
title: Coupler lookup — flux-noise diagnosis of the manipulate T2
date: 2026-05-01
experiment: 260420_T2_AC_stark
notebook: coupler_lookup_analysis.ipynb
data: D:\experiments\260420_T2_AC_stark\data\coupler_lookup_tables.pkl
tags: [QEC, coupler, manipulate, flux-noise, dephasing, Tphi, 1f]
---

# Coupler lookup — flux-noise diagnosis of the manipulate T2

## 1. System and goal

We have a three-mode unit:

- a **manipulate** cavity at $\omega_m^{\rm bare} \approx 4977$ MHz — the protected mode whose coherence we want to understand,
- a **transmon coupler** at $\omega_c(\Phi) \approx 4.09$–$4.72$ GHz — the *only* flux-tunable element in the chain,
- a transmon **qubit** with $g\!\leftrightarrow\!f$ at $\omega_{gf}$, used to convert flux-knob current $I$ into the dressed manipulate frequency through the f0g1 sideband: $f_{0g1}(I) = \omega_{gf} - \omega_m^{\rm dressed}(I)$.

The coupler is connected to the manipulate by a single dispersive coupling $g$. Tuning $\omega_c$ pulls $\omega_m^{\rm dressed}$ via the Lamb shift

$$
\omega_m^{\rm dressed}(\Phi) \;=\; \omega_m^{\rm bare} \;+\; \frac{g^2}{\omega_m^{\rm bare}-\omega_c(\Phi)} \,.
$$

**Question.** Is the manipulate $T_2^{\rm Ramsey}$ limited by 1/f flux noise? If so, the only plausible source is the coupler (the only flux-sensitive element). The diagnostic is to invert the flux-noise dephasing formula on both qubits and compare the implied $A_\Phi$ — they should agree.

They do not. They disagree by **~2 orders of magnitude**, which is the central puzzle of this report.

---

## 2. What was measured

For each setpoint current $I \in [-0.15, +0.40]$ mA (18 points, 11 fully characterized) the lookup pickle contains:

- $f_{0g1}(I)$ (sideband-from-the-qubit) and $f_c(I)$ (direct coupler spectroscopy) — both fit globally as polynomials,
- coupler $T_1$ from `T1CouplerExperiment`,
- coupler $T_2$ — extracted from the **spec-vs-power FWHM intercept**: Lorentzian linewidth at several drive amplitudes, then ${\rm FWHM}^2 = b + m A^2$ → $T_2 = 1/(\pi b)$. (All swept points fall in the `short_T2` regime; no Ramsey was usable on the coupler at any flux in this run.)
- manipulate $T_1$, $T_2^{\rm Ramsey}$, $T_2^{\rm echo}$ from standard sequences on the storage cavity.

Flux/current is anchored as $\Phi/\Phi_0 = 0$ at $I = -0.21$ mA, $\Phi/\Phi_0 = 0.5$ at $I = 0.75$ mA, giving $1\Phi_0 = 1.92$ mA.

---

## 3. Frequency map and Lamb fit

The polynomials track $f_{0g1}(I)$ and $f_c(I)$ cleanly across the swept range. Using $\omega_m^{\rm dressed}(I) = \omega_{gf} - f_{0g1}(I)$ and fitting the dispersive form against $f_c(I)$:

$$
\boxed{\ g = 100.05 \pm 0.32~\text{MHz}, \qquad \omega_m^{\rm bare} = 4977.10 \pm 0.18~\text{MHz}\ }
$$

(2-level model; the transmon-corrected form with $\alpha\!\approx\!100$ MHz is available as a toggle but unused here.)

The dispersive parameter $|\Delta|/g$ where $\Delta = \omega_m^{\rm bare}-\omega_c(\Phi)$ falls below 5 in
$$\Phi/\Phi_0 \in [-0.021,\, +0.178] \quad (I \in [-0.25,\, +0.13]~\text{mA})\,.$$
Inside this band the second-order Lamb expansion has $\gtrsim 4\%$ relative error and any quantity derived from $\partial f_m/\partial\Phi$ should not be over-interpreted. **All numerical comparisons below are taken outside the band wherever the data permits.**

---

## 4. Coherence summary

### 4.1 Coupler

| quantity            | range                     | comment                              |
|---------------------|---------------------------|--------------------------------------|
| $T_1^{c}$           | 44 – 84 μs                | mostly flat in $\Phi$                |
| $T_2^{c}$ (spec-FWHM)| 25 – 180 ns              | strongly $\Phi$-dependent            |
| $\partial f_c/\partial\Phi$ | $-490$ to $-7450$ MHz/$\Phi_0$ | from polynomial          |

$T_2^{c}$ scales roughly with $1/(\partial f_c/\partial\Phi)$ — a factor 15 in $|df_c/d\Phi|$ produces a factor ~7 in $T_2^c$. That sub-linearity is the first sign that something other than a single 1/f source is in play, but the trend is qualitatively the right shape for flux-driven dephasing.

### 4.2 Manipulate

| quantity                 | range          |
|--------------------------|----------------|
| $T_1^m$                  | 134 – 260 μs   |
| $T_2^{\rm Ramsey}$       | 78 – 289 μs    |
| $T_2^{\rm echo}$         | 93 – 379 μs    |
| $\partial f_m/\partial\Phi$ (Lamb model) | $\sim 50$ – $150$ MHz/$\Phi_0$ |

$T_2^{\rm Ramsey}$ improves *monotonically* away from the resonance band: from ~78 μs near $\Phi/\Phi_0\!\approx\!0.03$ up to ~290 μs at $\Phi/\Phi_0\!\approx\!0.32$. $T_\varphi = (1/T_2^R - 1/(2T_1))^{-1}$ varies from ~110 μs to ~640 μs. Echo gives a consistent picture and is roughly $1.3$–$1.5\times$ the Ramsey number, i.e., the dephasing has substantial low-frequency content (the gap is bigger than the typical pure-Lorentzian Ramsey/echo ratio of 1).

That **Ramsey/echo ratio < 2** plus monotonic $T_\varphi(\Phi)$ is consistent with 1/f-style noise driving the manipulate dephasing, modulated by $\partial f_m/\partial\Phi$.

---

## 5. The 1/f inversion on each qubit

Using the standard Ramsey-1/f form

$$
\frac{1}{T_\varphi} \;=\; 2\pi\,\bigl|\partial f/\partial\Phi\bigr|\;A_\Phi\;\sqrt{2\,|\ln(\omega_{ir}\,t)|}, \qquad \omega_{ir} = 2\pi/\tau_{\rm meas},
$$

and inverting at each measured point:

| qubit       | median $A_\Phi$        | range across $\Phi$        |
|-------------|------------------------|----------------------------|
| coupler     | **~200 μΦ₀**           | 160 – 280 μΦ₀ (factor 1.7) |
| manipulate  | **~1.5 μΦ₀**           | 0.9 – 3.4 μΦ₀ (factor 4)   |

Outside the dispersive-breakdown band the manipulate's $A_\Phi$ is even tighter: $\sim 0.9$–$1.4\,\mu\Phi_0$, well within typical transmon flux-noise expectations.

The ratio $A_\Phi^m/A_\Phi^c \approx 0.008$ — flat at ~1/125 across all measured flux points. This persists everywhere, including outside the breakdown band.

> [!warning] If a single 1/f flux source drove both qubits' dephasing, this ratio would be $1.0$.

---

## 6. Reading the discrepancy

Several explanations are possible. The flux/current calibration cancels in the ratio, so it is not a $\Phi_0$-axis problem. The Lamb-shift model affects $\partial f_m/\partial\Phi$ but not $\partial f_c/\partial\Phi$; both scale identically with the calibration. So the 100× has to live in either (a) what each "T2" actually measures or (b) which dephasing channel each qubit actually sees.

### 6.1 Most likely: the coupler T2 is NOT a Ramsey $T_\varphi$

The coupler T2 is extracted from the **Lorentzian-fit FWHM intercept of spec vs power**. This conversion equates linewidth with $1/(\pi T_\varphi)$ — strictly true only for *Markovian* (white) dephasing. Under 1/f flux noise the line is closer to Gaussian and the FWHM-to-$T_\varphi$ relation picks up a model-dependent geometric factor of order unity (not 100×, so this alone does not close the gap).

The bigger systematic is that **spectroscopy linewidth integrates noise across a much wider band than Ramsey does**. Ramsey at $t = T_2$ filters in noise around $1/t \sim 1/(80~\mu s) \sim 12$ kHz; spec-vs-power FWHM at fixed averaging time samples the slow drift between sweeps, on timescales of $\sim 1/(t_{\rm point})$ down to $1/\tau_{\rm sweep}$ — typically seconds to ten-of-seconds, i.e., low-frequency 1/f tails. For 1/f noise the integrated power is logarithmic in this bandwidth, so the discrepancy from log-window mismatch alone is at most a factor $\sim 3$, again not 100×.

What **does** plausibly explain a factor 100 is that the coupler's spec linewidth is dominated by **non-flux** broadening:
- TLS-induced spectral diffusion — flat-ish across $\Phi$, would mimic constant $A_\Phi^c$;
- charge dispersion at the coupler's $E_J/E_C$ (not measured directly here);
- power broadening that survives the FWHM² intercept (e.g., if the line is not Lorentzian, the intercept is biased);
- residual photon shot noise from the strongly-coupled manipulate ($\chi_{mc}\!\sim\!g^2/\Delta^3 \cdot \ldots$) and from the readout/conditioning drives.

The signature that *would* discriminate: $A_\Phi^c$ should be exactly flat in $\Phi$ if 1/f flux noise dominates the coupler. Observed $A_\Phi^c$ varies by factor 1.7 — small, but trending downward toward higher $|df_c/d\Phi|$, which is the *wrong* direction for a flat noise floor and the *right* direction for a constant non-flux contribution to the linewidth. **This is consistent with the coupler linewidth being a sum: a true 1/f flux part plus a ~constant non-flux floor of order MHz.** At low $|df_c/d\Phi|$ the floor dominates and $A_\Phi^c$ is over-estimated.

### 6.2 Possibility: the manipulate is *not* dephased by coupler flux noise at all

If the manipulate's $T_\varphi$ is set by some other channel — photon shot noise from a hot bath, intrinsic dielectric loss converted to dephasing through some non-radiative process, drive leakage — then the manipulate-derived $A_\Phi^m$ is meaningless: we are dividing a non-flux dephasing rate by a flux derivative.

Two checks against this:
- $A_\Phi^m$ is roughly **flat** in $\Phi$ outside the breakdown band ($\sim 1\,\mu\Phi_0$), exactly the signature 1/f flux noise predicts. A non-flux-driven $T_\varphi$ would generally give a *non-flat* implied $A_\Phi^m \propto 1/(df_m/d\Phi)$.
- Echo/Ramsey ratio is $\sim 1.3$–$1.5$, indicating substantial low-frequency content, again consistent with 1/f.

So the **manipulate side of the inversion looks self-consistently 1/f-flux-noise-like, with $A_\Phi \approx 1\,\mu\Phi_0$**. The coupler side does not.

### 6.3 Geometry / participation: can the noise sources be physically separate?

Two qubits on the same chip can in principle see different 1/f flux baths — TLS spins distributed across the substrate near each junction, with very different effective "loops" enclosing them. But the only loop in this system that is *deliberately* flux-coupled is the SQUID of the coupler, and the manipulate's flux sensitivity is engineered to come *only* through the dispersive Lamb channel. If both qubits saw the *same* TLS-spin bath but with different geometric prefactors (different $A_\Phi^{\rm geom}$), the ratio would be a fixed geometry constant — not necessarily 1.

This is the only "boring" explanation that survives: **the manipulate's flux sensitivity is dominated by something other than the dispersive coupling to the coupler** — e.g., direct stray inductive coupling to the flux line. To match $A_\Phi^c \approx 200\,\mu\Phi_0$ via the manipulate's measured $T_\varphi$, the manipulate's true $\partial f_m/\partial\Phi$ would need to be ~125× larger than the Lamb-model prediction — i.e., $\sim 10$ GHz/$\Phi_0$, comparable to the coupler itself. That is implausible: independent measurement of $f_m(I)$ via $f_{0g1}$ would have already shown such a swing, and it does not.

So this scenario can be ruled out by data — which leaves §6.1 as the most likely culprit: **the coupler's spec-FWHM "T2" overstates the flux-noise contribution, because the linewidth has a sizeable non-flux component.**

---

## 7. Flaws and improvements in the experimental flow

### Methodological

1. **Coupler T2 is never compared apples-to-apples with a Ramsey number.** Every swept point lands in the `short_T2` regime, so the only handle on coupler $T_\varphi$ is spec-FWHM. Without at least one cross-check at a flux where Ramsey works (high-flux end, lowest $|df_c/d\Phi|$), the spec→$T_\varphi$ conversion is uncalibrated. **Recommendation:** push to slightly higher current and try a Ramsey on the coupler near $|df_c/d\Phi|\!\sim\!500$ MHz/$\Phi_0$. Even one anchor point lets us back out the conversion factor.

2. **Lorentzian model for spec vs power.** The FWHM² vs A² linearization implicitly assumes a Lorentzian (Markovian) line. Under 1/f-dominated dephasing the line is Gaussian; the Lorentzian fit returns a biased FWHM that is not $1/(\pi T_\varphi)$. **Recommendation:** also fit a Voigt or pseudo-Voigt to the spec lines and report the Gaussian width separately — that width tracks the static-detuning distribution, which is the meaningful quantity in the 1/f limit.

3. **Sparse polynomial fit drives $\partial f_m/\partial\Phi$.** $f_c(I)$ is a deg-3 polynomial through 15 seed points; $f_{0g1}(I)$ a deg-5 polynomial through 10. The numerical derivative through the Lamb model is therefore noisy at the edges. **Recommendation:** add 3–5 points at the high-current end (where $|\Delta|/g$ is largest and the inversion is cleanest) before drawing quantitative conclusions.

4. **Two-level Lamb model only.** `USE_TRANSMON_CORRECTION=False`. With $\alpha\sim 100$ MHz, the multilevel form $g^2\alpha/[\Delta(\Delta+\alpha)]$ shifts both $g$ and the inferred $\partial f_m/\partial\Phi$. **Recommendation:** measure the coupler anharmonicity at a few flux points and re-run the fit — or at least show that the conclusions are robust to the toggle.

5. **Single log-factor used for both qubits.** Section 5/6 use the same $\sqrt{2|\ln(\omega_{ir}t)|}$ on both sides with $\tau_{\rm meas} = 3600$ s. For Ramsey $t \approx T_2^R$, that's correct. For spec-vs-power, the relevant "t" is more like the per-point integration time, and the IR cutoff is the full sweep duration. They differ. **Recommendation:** evaluate the log-factor mismatch explicitly; even with a generous correction it does not close 100×, but it is currently glossed over.

6. **Implicit assumption that coupler T2 noise floor is flux-driven 1/f.** $A_\Phi^c$ trends with $\Phi$ rather than being flat — that is *evidence against* the assumption, but the notebook reports a "median $A_\Phi^c$" as if it were a fitted constant. **Recommendation:** fit $1/T_2^c$ as $a + b\,|df_c/d\Phi|$ instead of pure proportionality. The intercept $a$ quantifies the non-flux floor; the slope $b$ gives the true $A_\Phi^c$.

### Logical / framing

7. **The dispersive-breakdown band is shaded in the plots but not explicitly excluded from the median.** Currently the "median $A_\Phi^m = 1.52\,\mu\Phi_0$" includes points inside the band, where the model can be off by tens of percent. **Recommendation:** report a "band-excluded" median alongside.

8. **No echo-based cross-check on the manipulate's flux-noise hypothesis.** The Ramsey/echo ratio is $\sim 1.3$–$1.5$; a true 1/f source predicts a specific ratio (close to 1 in the slow-noise limit, growing toward 2 as the spectrum flattens). **Recommendation:** do the explicit Ramsey-vs-echo comparison for one or two flux points and check whether the implied 1/f amplitude is consistent across both.

9. **No independent test that the manipulate flux sensitivity comes from the dispersive channel.** Could be checked by purposely detuning the coupler to a known $\Delta$ via DC bias and watching $\partial f_m/\partial\Phi$ collapse as $1/\Delta^2$ — the Lamb fit predicts a specific dependence. The current data set does this implicitly, but the residuals should be plotted (Lamb-model residual vs $\Phi$) to confirm there is no additional flux channel hidden in the data.

---

## 8. Bottom line

- **Manipulate $T_2^{\rm Ramsey}$ is consistent with 1/f flux noise** with $A_\Phi^m \approx 1\,\mu\Phi_0$ (flat in $\Phi$, Ramsey/echo ratio $<2$).
- The **coupler dephasing measurement is not clean enough** to ascribe to a single flux source. Spec-FWHM-derived $A_\Phi^c \approx 200\,\mu\Phi_0$ is implausibly large compared to the manipulate inversion *and* shows a flux dependence that argues for a non-flux contribution to the linewidth.
- The most defensible reading: there *is* a 1/f flux noise floor of order $1\,\mu\Phi_0$ shared by both modes; the coupler additionally suffers a non-flux dephasing channel (TLS, photon shot noise, charge dispersion, or systematic in the FWHM intercept) of order MHz that swamps its flux contribution everywhere we have data.
- The cleanest one-experiment test of this hypothesis is a **Ramsey on the coupler at low $|df_c/d\Phi|$**; the next cleanest is **fitting $1/T_2^c$ as $a + b\,|df_c/d\Phi|$** on the existing data.

The 100× discrepancy is not telling us the manipulate is dephased by something exotic. It is telling us that the coupler's spec-FWHM "T2" is not a $T_\varphi$.
