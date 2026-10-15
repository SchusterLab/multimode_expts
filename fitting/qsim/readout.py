"""Optional two-cloud IQ refit from Jonginn's Sept 28 analysis.

This cannot repair acquisition errors caused by stale active-reset thresholds.
Ground/excited labels assume the saved contrast sign remains correct.
"""

def fit_readout_clouds(iq, saved_contrast, max_samples=50000, max_iter=300,
                      tol=1e-7, min_weight=0.02, min_separation=2.0):
    """Fit two unlabeled IQ clouds, once per job; never fit the time trace.

    Labels ASSUME the saved sign of Ie-Ig is still correct. The weight and
    separation cuts are numerical/model usability guards, not confidence levels.
    Two Gaussians do not establish that leakage, drift or preparation errors are
    absent. Inputs and outputs use the same IQ coordinates (no ADC correction).
    """
    import numpy as np

    points = np.asarray(iq, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 100:
        raise ValueError('Readout fit needs at least 100 paired I/Q shots')
    if not np.isfinite(points).all() or not np.isfinite(saved_contrast):
        raise ValueError('Readout fit received non-finite IQ or saved contrast')
    if saved_contrast == 0:
        raise ValueError('Saved Ie-Ig is zero; ground/excited labels are unknown')
    if max_samples < 100 or max_iter < 2 or tol <= 0:
        raise ValueError('Invalid readout-fit sample/iteration/convergence settings')
    if not 0 < min_weight < 0.5 or min_separation <= 0:
        raise ValueError('Invalid readout-fit weight/separation guards')
    input_count = len(points)
    if len(points) > max_samples:
        indices = np.random.default_rng(0).choice(len(points), max_samples, replace=False)
        points = points[indices]
    origin = points.mean(axis=0)
    scale = np.sqrt(np.mean((points - origin)**2))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        raise ValueError('Readout shots have no usable IQ variation')
    z = (points - origin) / scale
    global_cov = np.array([[np.mean(z[:, 0]**2), np.mean(z[:, 0]*z[:, 1])],
                           [np.mean(z[:, 0]*z[:, 1]), np.mean(z[:, 1]**2)]])
    # Scalar ridge in standardized coordinates; no matrix inversion/BLAS needed.
    ridge = 1e-6
    principal_angle = 0.5*np.arctan2(2*global_cov[0, 1], global_cov[0, 0]-global_cov[1, 1])
    starts = [np.array([1., 0.]), np.array([np.cos(principal_angle), np.sin(principal_angle)])]
    best = None
    for direction in starts:
        projection = z[:, 0]*direction[0] + z[:, 1]*direction[1]
        cuts = np.quantile(projection, [0.25, 0.75])
        centers = np.array([z[projection <= cuts[0]].mean(axis=0),
                            z[projection >= cuts[1]].mean(axis=0)])
        cov = np.repeat((global_cov/2 + ridge*np.eye(2))[None], 2, axis=0)
        weights, previous, converged = np.array([0.5, 0.5]), -np.inf, False
        for iteration in range(max_iter):
            logp = np.empty((len(z), 2))
            valid = True
            for k in range(2):
                a, b, c = cov[k, 0, 0], cov[k, 0, 1], cov[k, 1, 1]
                determinant = a*c - b*b
                if determinant <= 0 or not np.isfinite(determinant):
                    valid = False
                    break
                x, y = z[:, 0]-centers[k, 0], z[:, 1]-centers[k, 1]
                quadratic = (c*x*x - 2*b*x*y + a*y*y)/determinant
                logp[:, k] = np.log(weights[k]) - np.log(2*np.pi) - 0.5*(np.log(determinant)+quadratic)
            if not valid:
                break
            peak = logp.max(axis=1)
            probability = np.exp(logp - peak[:, None])
            total = probability.sum(axis=1)
            likelihood = np.mean(peak + np.log(total))
            if not np.isfinite(likelihood) or likelihood < previous - 1e-6:
                break
            if np.isfinite(previous) and abs(likelihood-previous) <= tol*(1+abs(previous)):
                converged = True
                break
            previous = likelihood
            responsibility = probability / total[:, None]
            counts = responsibility.sum(axis=0)
            weights = counts / len(z)
            if np.min(weights) < min_weight:
                break
            for k in range(2):
                r = responsibility[:, k]
                centers[k] = np.sum(r[:, None]*z, axis=0)/counts[k]
                x, y = z[:, 0]-centers[k, 0], z[:, 1]-centers[k, 1]
                a = np.sum(r*x*x)/counts[k] + ridge
                b = np.sum(r*x*y)/counts[k]
                c = np.sum(r*y*y)/counts[k] + ridge
                cov[k] = [[a, b], [b, c]]
        if converged and (best is None or likelihood > best[0]):
            best = (likelihood, centers.copy(), cov.copy(), weights.copy(), iteration+1)
    if best is None:
        raise ValueError('Two-cloud readout fit failed to converge or a component vanished')
    likelihood, centers, cov, weights, iterations = best
    centers, cov = origin + scale*centers, scale**2*cov
    order = np.argsort(centers[:, 0])
    if saved_contrast < 0:
        order = order[::-1]
    centers, cov, weights = centers[order], cov[order], weights[order]
    delta = centers[1] - centers[0]
    distance = np.sqrt(np.sum(delta**2))
    if abs(delta[0]) <= 1e-6*scale or distance <= 1e-6*scale:
        raise ValueError('Fitted clouds cannot be labeled from the saved I contrast sign')
    direction = delta/distance
    variances = (direction[0]**2*cov[:, 0, 0] + 2*direction[0]*direction[1]*cov[:, 0, 1]
                 + direction[1]**2*cov[:, 1, 1])
    separation = distance / np.sqrt(np.mean(variances))
    if not np.isfinite(separation) or separation < min_separation:
        raise ValueError(f'Readout clouds overlap too much: separation={separation:.2f} < {min_separation:g}')
    projected = centers[:, 0]*direction[0] + centers[:, 1]*direction[1]
    return dict(centers=centers, covariance=cov, weights=weights,
                angle_deg=float(-np.degrees(np.arctan2(delta[1], delta[0]))),
                Ig=float(projected[0]), Ie=float(projected[1]),
                projected_sigma=np.sqrt(variances), separation=float(separation),
                converged=True, iterations=iterations, mean_log_likelihood=float(likelihood-2*np.log(scale)),
                sample_count=len(points), input_count=input_count,
                label_assumption='ground/excited ordered by the saved sign of Ie-Ig')
