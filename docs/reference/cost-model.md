# Cost model reference

flopscope bills compute as:

```
charged = int(flop_cost × weight)
```

`flop_cost` is the entire analytical FLOP count for a call (shape-dependent).
`weight` is a per-element tier factor that converts analytical FLOPs to an
equivalent billing unit.  The two concerns are kept separate: `flop_cost`
carries all shape constants; `weight` carries only the per-element tier.

For hardware calibration — how weights are measured and what empirical
values differ from the declared tier — see
[docs/reference/empirical-weights.md](empirical-weights.md).

---

## Conventions (declared layer)

### FMA=2

Each floating-point multiply, add, subtract, divide, or square root counts
as 1 FLOP.  A fused multiply-add (FMA) therefore counts as 2.  This matches
the textbook convention used in Golub & Van Loan, _Matrix Computations_, 4th
ed. (G&VL 4e) §1.1.  All formulas in this document are stated in FMA=2 units
unless noted.

### Comparison and select

A single comparison (`>`, `==`, `!=`, …) or conditional-select (`where`,
`choose`) counts as 1 FLOP.  Sorting, partition, and percentile operations
use this convention when counting per-element work.

### Transcendental tier (weight 16.0)

Operations whose per-element cost is dominated by a libm minimax polynomial
evaluation (sin, cos, tan, exp, log, arcsin, arccos, arctan, arcsinh,
arccosh, arctanh, power, and their NumPy 2.x aliases) are billed at weight
16.0.  The `flop_cost` formula is `numel(output)` (1 per element); the 16×
factor is supplied entirely by the weight.

Citation: J.-M. Muller, _Elementary Functions_, 3rd ed., Birkhäuser (2016),
Chapter 2 (range-reduction + minimax polynomial).

A subset of moderate-cost binary ops (floor_divide, mod/remainder, fmod,
arctan2, hypot, logaddexp, logaddexp2) is calibrated into the same tier
(weight 16.0).  See empirical-weights.md for measured values.

### Half-tier transcendentals (weight 8.0)

Not currently used in the active weight table; reserved for future ops.

### Gather tier (weight 4.0)

Indexing and branch-heavy per-element ops (gather, scatter, where, compress,
extract, choose, take, place, putmask) charge `numel(input)` at weight 4.0.
The weight reflects the non-trivial branch or index computation per element.

### Views and metadata (weight 0.0)

Operations that return a view of existing memory, or that inspect metadata
without touching element values, are billed 0.  Examples: reshape, ravel,
flatten, transpose, diagonal (as a view), squeeze, broadcast_to, astype (no
copy), fftshift/ifftshift, fftfreq/rfftfreq, linalg.diagonal,
linalg.matrix_transpose.

### Composite ops (weight 1.0 with heterogeneous flop_cost)

When an operation mixes sub-tiers internally (e.g. random samplers, stats
kernels, norms with SVD), all per-element factors are folded into `flop_cost`
and the active weight is set to 1.0.  This avoids double-counting with the
tier factor.

### Weight tier policy

The tier assignments are CI-enforced: `tests/test_weight_tier_policy.py`
asserts that every registered op's active weight belongs to one of the
declared tiers (0.0, 1.0, 2.0, 4.0, 8.0, 16.0) and that the tier matches
the op's family classification.

### NumPy 2.x ufunc aliases

NumPy 2.x introduced `acos`, `acosh`, `asin`, `asinh`, `atan`, `atanh`,
`atan2`, `pow`, and `divmod` as canonical aliases for their `arc*` /
`power` / `floor_divide` counterparts (identical ufunc objects).  flopscope
resolves these via `_UFUNC_ALIAS_RENAMES` in `_weights.py` so each alias
charges the same weight as its canonical twin.  The fix was introduced in
commit `7f0b0a18`.

---

## Per-family tables

The families below cover the 602 registered ops.  Where an entire family
shares one formula the rule is stated once; only exceptions are tabulated.

---

### Elementwise (pointwise unary and binary)

**Family rule**: `flop_cost = numel(output)`.

**Baseline tier (weight 1.0)**: arithmetic (+, −, ×, ÷, √), rounding
(ceil, floor, trunc, rint, around/round), sign/abs, logical (not, and, or,
xor), bitwise (and, or, xor, invert, left_shift, right_shift), comparisons
(equal, not_equal, greater, less, greater_equal, less_equal), copies
(positive, negative, real, imag, conj/conjugate, fabs, modf, frexp, spacing,
nan_to_num, isclose, isneginf, isposinf, deg2rad/degrees, rad2deg/radians,
ldexp, nextafter, copysign, heaviside, signbit), and their NumPy aliases.

**Transcendental tier (weight 16.0)**: exp, exp2, expm1, log, log2, log10,
log1p, cbrt, sin, cos, tan, sinh, cosh, tanh, arcsin, arccos, arctan,
arcsinh, arccosh, arctanh, sinc, i0, power, angle, and their NumPy 2.x
aliases (asin, acos, atan, asinh, acosh, atanh, atan2, pow).

**Moderate binary tier (weight 16.0)**: arctan2/atan2, hypot, logaddexp,
logaddexp2, floor_divide, mod/remainder, fmod, float_power.

**Basis**: DECLARED per-element FMA=2 convention and empirical calibration.
Source: `src/flopscope/_pointwise.py`.

---

### Reduction

**Family rule**: `flop_cost = numel(input) − numel(output)` (orbit-mapping
model; one add or compare per element consumed by the reduction).

Ops that do more than one accumulation pass carry the extra passes in
`flop_cost` (never in the weight column): the variance family makes four
passes (mean-sum, centre, square, variance-sum), `ptp` makes two (max + min)
plus the per-output subtract, and `mean`/`average` add the per-output divide.

| Op | flop_cost | weight | basis |
|---|---|---|---|
| `sum`, `prod`, `max`, `min`, `any`, `all`, `cumsum`, `cumprod`, `nansum`, `nanmax`, `nanmin`, `nanprod`, `nancumsum`, `nancumprod`, `cumulative_sum`, `cumulative_prod` | numel(input) − numel(output) | 1.0 | DECLARED reduction skeleton (one add per consumed element) |
| `mean`, `average` (unweighted) | numel(input) | 1.0 | DERIVED: reduction (numel−M) + M divides |
| `average(weights=)` | numel − M + 2·numel + M | 1.0 | DERIVED: a·w pass + a·w sum + w sum + M divides |
| `std`, `var`, `nanstd`, `nanvar` | ≈ 4 × numel(input) (std: + M sqrt) | 1.0 | DERIVED four-pass: mean-sum, centre, square, var-sum (exact: 2·numel + 2·(numel−M) + 2M) |
| `argmax`, `argmin` | numel(input) | 1.0 | DECLARED scan |
| `median`, `nanmedian` | axis length per output slice | 1.0 | DECLARED; partition (introselect) per output |
| `percentile`, `nanpercentile`, `quantile`, `nanquantile` | axis length per output slice | 1.0 | DECLARED; partition (introselect) per output |
| `ptp` | 2 × numel(input) − numel(output) | 1.0 | DERIVED: max pass + min pass + M subtracts (2·(numel−M)+M) |
| `count_nonzero` | numel(input) | 1.0 | DECLARED comparison scan (every element tested regardless of axis) |
| `nanmean` | numel(input) | 1.0 | DERIVED: reduction (numel−M) + M divides; billed identically to mean |

Source: `src/flopscope/_pointwise.py`; reduction accumulation model in
`src/flopscope/_accumulation/`.

---

### Contraction (einsum family)

**Family rule** (DERIVED, G&VL 4e §1.1.11):

```
flop_cost = (K − 1) × M_unique + M_unique
           = (2K − 1) × M_unique
```

where `K` = product of contracted-axis dimensions, `M_unique` = number of
output cells actually computed (equals `prod(output dims)` for non-aliased
inputs; reduced to the unique-orbit count when the output has symmetry, e.g.
`A @ A` or `outer(v, v)`).

For a plain `(m, k) @ (k, n)` matmul: `flop_cost = 2mkn − mn`.

Multi-operand einsum (`k ≥ 3`) walks the `opt_einsum` optimal binary path and
sums per-step costs.

| Op | flop_cost formula | basis | source |
|---|---|---|---|
| `matmul`, `linalg.matmul` | `2mkn − mn` | DERIVED | G&VL 4e §1.1.11 |
| `dot` | `(2K−1)×M_out`; matrix-vector = `m(2k−1)` | DERIVED | G&VL 4e §1.1.11 |
| `inner` | `(2K−1)×M_unique`; aliased `inner(A,A)` → `n(n+1)/2` output cells | DERIVED | G&VL 4e §1.1.11 |
| `outer`, `linalg.outer` | `m×n` (K=1, one multiply per output cell) | DERIVED | G&VL 4e §1.1.1 |
| `tensordot`, `linalg.tensordot` | `(2K−1)×M_out` via einsum subscript path | DERIVED | G&VL 4e §1.1.11 |
| `vdot`, `vecdot`, `linalg.vecdot` | `2N − 1` | DERIVED | G&VL 4e §1.1.2 |
| `matvec`, `vecmat` | `m(2k−1)` | DERIVED | G&VL 4e §1.1.8 |
| `einsum` | whole-expression accumulation; k≥3 binary path | DERIVED | G&VL 4e §1.1.11; `_accumulation/_cost.py` |
| `kron` | `a.size × b.size` (outer product, no contraction) | DERIVED | Kronecker product definition; FMA=2 |
| `linalg.matrix_power` | `(⌊log₂ k⌋ + popcount(k) − 1) × matmul_cost(n,n,n)` | DERIVED | Knuth TAOCP §4.6.3 × G&VL 4e §1.1.11 |
| `linalg.multi_dot` | sum of optimal-chain matmul costs (CLRS §15.2); each step uses `matmul_cost(m,k,n)` = `2mkn − mn` | DERIVED | G&VL 4e §1.1.11; `_compound.py:multi_dot_cost` |

All contraction ops use **weight 1.0** (the shape constants capture everything).
Source: `src/flopscope/_accumulation/`, `src/flopscope/_flops.py`.

---

### Generator (linspace, arange, and kin)

| Op | flop_cost | basis | source |
|---|---|---|---|
| `arange` | `2 × numel(output)` | DERIVED: `start + i×step` per element = 1 mul + 1 add (FMA=2) | `_free_ops.py`; numpy arraytypes.c.src |
| `linspace` | `2 × numel(output)` (handles broadcast start/stop and `retstep=True`) | DERIVED: same affine model as arange | `_free_ops.py`; commit 790d19af + retstep fix |
| `geomspace` | `2 × num × B + 6B` where `B` = product of broadcast batch dims | DERIVED: log + linspace + exp per-batch per-point | `_free_ops.py` |
| `logspace` | same as geomspace | DERIVED | `_free_ops.py` |
| `zeros`, `ones`, `full`, `zeros_like`, `ones_like`, `full_like`, `eye`, `identity`, `empty`, `empty_like` | 0 (allocation, no arithmetic) | DECLARED free/metadata | `_free_ops.py` |
| `meshgrid` | dense: `len(xi) × prod(sizes)`; sparse (`sparse=True`): `sum(sizes)`; `copy=False` views: 1 | DECLARED: numel of materialized output grids | `_free_ops.py` |

Weight: **1.0** for all counted generators.  Source: `src/flopscope/_free_ops.py`.

---

### Sort and select

**Family rule** (DECLARED, Knuth TAOCP v3 §5.2):

| Op | flop_cost | basis |
|---|---|---|
| `sort`, `argsort` | `num_slices × n × ⌈log₂ n⌉` | DECLARED comparison sort (n = axis length) |
| `unique`, `unique_counts`, `unique_inverse`, `unique_values`, `unique_all` | `n × ⌈log₂ n⌉` (axis=None); `num_slices × shape[axis] × ⌈log₂ shape[axis]⌉` (axis=k) | DECLARED sort-based; axis-aware per-slice |
| `lexsort` | `k × n × ⌈log₂ n⌉` (k = number of keys, n = sequence length) | DECLARED |
| `partition`, `argpartition` | `num_slices × n × len(kth)` | DECLARED quickselect O(n) expected |
| `searchsorted` | `m × ⌈log₂ n⌉` (m = queries, n = sorted size) | DECLARED binary search |
| `sort_complex` | `a.size × ⌈log₂(a.size)⌉` on flattened size | DECLARED |
| `in1d`, `isin` | `(n + m) × ⌈log₂(n + m)⌉` (sort path); `max(sort_cost(n+m), 2nm)` when numpy's masked-loop path triggers (small integer ar2) | DECLARED algo-aware |
| `intersect1d`, `setdiff1d`, `setxor1d`, `union1d` | `(n + m) × ⌈log₂(n + m)⌉` | DECLARED |

All sort/select ops use **weight 1.0**; comparison = 1 FLOP convention.
Source: `src/flopscope/_sorting_ops.py`, `src/flopscope/_flops.py` (`sort_cost`, `search_cost`).

---

### Linalg direct (non-iterative)

All ops use **weight 1.0** with all shape constants in `flop_cost`.  Per-matrix
cost is multiplied by the batch dimension product for stacked inputs.  Zero-dim
matrices charge 0.

| Op | flop_cost (per matrix) | basis | source |
|---|---|---|---|
| `linalg.cholesky` | `n³/3` | DERIVED: G&VL 4e Alg 4.2.1 (dpotrf); LAPACK dpotrf driver count | `_decompositions.py:cholesky_cost` |
| `linalg.qr` (reduced/complete) | `2(2mnk − 2k³/3)`, `k = min(m,n)` | DERIVED: G&VL 4e §5.2 (dgeqrf) + dorgqr Q-formation ≈ same count; LAWN 41 confirms | `_decompositions.py:qr_cost` |
| `linalg.qr` (r/raw) | `2mnk − 2k³/3` | DERIVED: factorization only | `_decompositions.py:qr_cost` |
| `linalg.solve` | `2n³/3 + 2n²×nrhs` | DERIVED: G&VL 4e §3.2 (dgesv = dgetrf + dgetrs) | `_solvers.py:solve_cost` |
| `linalg.inv` | `2n³` | DERIVED: G&VL 4e §3.4 (dgetrf + dgetri ≈ 2n³) | `_solvers.py:inv_cost` |
| `linalg.det` | `2n³/3 + n` | DERIVED: G&VL 4e §3.2 LU (dgetrf) + diagonal product | `_properties.py:det_cost` |
| `linalg.slogdet` | `2n³/3 + 18n` | DERIVED: LU (dgetrf) + sum of log\|diag\| (abs + 16/elem log + reduce) | `_properties.py:slogdet_cost` |
| `linalg.norm` (fro/L1/Linf) | `2 × numel(effective_shape) × n_groups` | DERIVED: FMA=2 square+accumulate or abs+accumulate | `_properties.py:norm_cost` |
| `linalg.norm` (ord=2, nuc) | `4 × m × n × min(m,n) × n_groups` | DERIVED: via SVD (4× baked in) | `_properties.py:norm_cost` |
| `linalg.vector_norm` | `2 × numel(effective_shape) × n_groups` (standard ord); `(18 × numel + 16) × n_groups` (general fractional p-norm: abs + pow per element) | DERIVED: FMA=2 | `_properties.py:vector_norm_cost` |
| `linalg.matrix_norm` | same as `linalg.norm` | DERIVED | `_properties.py` |
| `linalg.trace` | `min(m,n) × batch` | DERIVED: n−1 diagonal adds, batch-multiplied | `_properties.py:trace_cost` |
| `linalg.tensorinv` | `2n³`, `n = prod(shape[:ind])` | DERIVED: G&VL 4e §3.4 via inv | `_solvers.py:tensorinv_cost` |
| `linalg.tensorsolve` | `2n³/3 + 2n²`, `n = prod(shape[ind:])` | DERIVED: G&VL 4e §3.2 via solve | `_solvers.py:tensorsolve_cost` |
| `linalg.cond`, `linalg.matrix_rank` | `4 × m × n × min(m,n)` (via SVD) | DERIVED | `_properties.py` |
| `linalg.pinv`, `linalg.lstsq` | `m × n × min(m,n)` | DERIVED: LAPACK dgelsd / SVD path; G&VL 4e §5.5 | `_solvers.py` |
| `linalg.cross` | `6 × n` (delegates to `fnp.cross`) | DERIVED | `_aliases.py` |
| `linalg.multi_dot` | optimal chain matmul cost (CLRS §15.2); each step uses `matmul_cost(m,k,n)` = `2mkn − mn` | DERIVED | `_compound.py:multi_dot_cost` |
| `linalg.outer`, `linalg.tensordot`, `linalg.vecdot`, `linalg.matmul`, `linalg.matrix_power` | delegates to `fnp.*` | DERIVED | `_compound.py`, `_aliases.py` |
| `linalg.diagonal`, `linalg.matrix_transpose` | 0 (view) | DECLARED free | `_aliases.py` |

---

### Linalg iterative (eigen / SVD)

These ops use LAPACK drivers that iterate until convergence; counts are
leading-order estimates with confirmed-2026-06 citations.  All use
**weight 1.0**.  See the [Evidence appendix](#evidence-appendix-iterative-linalg-constants)
for the three-leg derivation.

| Op | flop_cost (per matrix) | basis | source |
|---|---|---|---|
| `linalg.eig` | `25n³` | DERIVED: G&VL 4e §7.5 (Hessenberg + Francis QR with eigenvectors); LAPACK Users' Guide Table 3.13 DGEEV-with-vectors = 26.33 n³ | `_decompositions.py:eig_cost` |
| `linalg.eigvals` | `10n³` | DERIVED: LAPACK Users' Guide Table 3.13 DGEEV values-only = 10.00 n³ (exact) | `_decompositions.py:eigvals_cost` |
| `linalg.eigh` | `9n³` | DERIVED: G&VL 4e §8.3 (dsyevd tridiagonalization + divide-and-conquer with eigenvectors) | `_decompositions.py:eigh_cost` |
| `linalg.eigvalsh` | `4n³/3` | DERIVED: G&VL 4e §8.3 (dsyevd tridiagonalization only, MRRR, no vectors) | `_decompositions.py:eigvalsh_cost` |
| `linalg.svd` (thin, full_matrices=False or square) | `6ab² + 20b³`, `a=max(m,n)`, `b=min(m,n)` | DERIVED: G&VL 4e §8.6 Table 8.6.1 R-SVD Σ+U₁+V (dgesdd thin path) | `_svd.py:svd_cost` |
| `linalg.svd` (full, full_matrices=True and m≠n) | `4a²b + 22b³` | DERIVED: G&VL 4e §8.6 Table 8.6.1 R-SVD full U (forming full m×m U dominates) | `_svd.py:svd_cost` |
| `linalg.svdvals` | `2ab² + 2b³` | DERIVED: G&VL 4e §8.6 Table 8.6.1 R-SVD Σ only (dgesdd values, no vectors) | `_decompositions.py:svdvals_cost` |
| `roots` | `10n³`, `n = len(p)−1` | DERIVED: companion-matrix eigvals (delegates to eigvals_cost) | `_polynomial.py`; note: uses raw `len(p)−1`, not stripped zero-padded length |

Per-matrix cost is multiplied by the batch dimension product.  Constants
marked "provisional": iteration counts are input-dependent and the cubic
constant is the standard textbook estimate.

---

### FFT

**Family rule** (DERIVED, Van Loan, _Computational Frameworks for the Fast
Fourier Transform_, 1992 §1.4, Cooley-Tukey radix-2):

| Op | flop_cost | basis |
|---|---|---|
| `fft.fft`, `fft.ifft`, `fft.fft2`, `fft.ifft2`, `fft.fftn`, `fft.ifftn` | `5 × N × ⌈log₂ N⌉`, `N = prod(transform dims)` | DERIVED: Van Loan 1992 §1.4; 5 real ops per butterfly |
| `fft.rfft`, `fft.irfft`, `fft.rfft2`, `fft.irfft2`, `fft.rfftn`, `fft.irfftn` | `5 × (N/2) × ⌈log₂ N⌉` | DERIVED: real-input / real-output half-spectrum |
| `fft.hfft` | `5 × (n_out/2) × ⌈log₂ n_out⌉` | DERIVED: hfft = irfft(conj(a)) — conjugate-symmetry halves the work (Van Loan 1992 §1.4) |
| `fft.ihfft` | `5 × (n/2) × ⌈log₂ n⌉` | DERIVED: same `hfft_cost(n)` formula |
| `fft.fftfreq`, `fft.rfftfreq`, `fft.fftshift`, `fft.ifftshift` | 0 | DECLARED free/metadata |

All counted FFT ops use **weight 1.0**.  Source: `src/flopscope/numpy/fft/_transforms.py`.

---

### Polynomial

| Op | flop_cost | basis | source |
|---|---|---|---|
| `polyval` | `2 × deg × points` (Horner: 1 mul + 1 add per coefficient per point, FMA=2) | DERIVED | `_polynomial.py` |
| `polyfit` | `2 × m × n × min(m,n)` (via least-squares SVD) | DERIVED | `_polynomial.py` |
| `polyadd`, `polysub` | `min(len_a, len_b)` | DERIVED | `_polynomial.py` |
| `polymul`, `convolve` (1-D full mode) | `2nm − n − m` (direct conv, FMA=2) | DERIVED; `convolve` always uses full-mode cost regardless of `mode=` argument | `_polynomial.py` |
| `polyder` | `t × n − t(t+1)/2`, `t = min(m, n−1)` (order-aware; one multiply per surviving coefficient per derivative step) | DERIVED | `_polynomial.py:polyder_cost` |
| `polyint` | `m × n + m(m−1)/2` (order-aware; m passes each dividing n+j coefficients) | DERIVED | `_polynomial.py:polyint_cost` |
| `roots` | `10n³`, `n = stripped companion dimension` (zero-leading/trailing coefficients stripped before companion matrix is built) | DERIVED: delegates to `eigvals_cost` on trimmed degree | `_polynomial.py:roots_cost` |

Source: `src/flopscope/_polynomial.py`.

---

### Random (module-level, Generator, RandomState)

Random ops are composite: the generation kernel cost and any setup cost
(PRNG state update, rejection sampling) are folded into `flop_cost` at
**weight 1.0**.

| Op / family | flop_cost | basis | source |
|---|---|---|---|
| `random.rand`, `random.uniform`, `random.random`, `random.random_sample`, `random.ranf`, `random.sample` | `numel(output)` | DECLARED: 1 FLOP per uniform draw | `_cost_formulas.py` |
| `random.randn`, `random.standard_normal`, `random.normal` | `6 × numel(output)` | DECLARED: Box-Muller / ziggurat ≈6 ops/sample (Devroye, _Non-Uniform Random Variate Generation_, 1986, §IV.4) | `_cost_formulas.py` |
| `random.randint`, `random.integers` | `numel(output)` | DECLARED | `_cost_formulas.py` |
| `random.choice` (replace=True, p=None) | `numel(output)` | DECLARED | `_cost_formulas.py` |
| `random.choice` (replace=True, p≠None) | `numel(output) + 3n + m×⌈log₂ n⌉` (n=population, m=size) | DERIVED: cumsum + normalize + searchsorted | `_cost_formulas.py`; confirmed issue audit |
| `random.choice` (replace=False, p=None) | `n` (Fisher-Yates O(n): conservative ceiling on tail-shuffle / Floyd's algorithm) | DECLARED | `_cost_formulas.py` |
| `random.choice` (replace=False, p≠None) | `sort_cost(n) = n × ⌈log₂ n⌉` (data-dependent rejection loop with weights) | DECLARED | `_cost_formulas.py` |
| `random.shuffle`, `random.permutation` | `numel(input)` | DECLARED: Fisher-Yates O(n) | `_cost_formulas.py` |
| `random.exponential` | `numel(output)` | DECLARED | `_cost_formulas.py` |
| `random.poisson`, `random.binomial`, `random.geometric`, `random.hypergeometric`, `random.negative_binomial`, `random.multinomial` | `numel(output)` | DECLARED | `_cost_formulas.py` |
| `random.multivariate_normal` | `d³/3 + 2×N×d² + 16×N×d` (N=size, d=dims) | DERIVED composite: covariance factorization + affine transform + N·d transcendental draws | `_cost_formulas.py` |
| `random.beta`, `random.dirichlet`, `random.f`, `random.gamma`, `random.gumbel`, `random.laplace`, `random.logistic`, `random.lognormal`, `random.logseries`, `random.pareto`, `random.power`, `random.rayleigh`, `random.standard_cauchy`, `random.standard_exponential`, `random.standard_gamma`, `random.standard_t`, `random.triangular`, `random.vonmises`, `random.wald`, `random.weibull`, `random.zipf` | `numel(output)` (with transcendental weight or composite constant, per distribution) | DECLARED / DERIVED | `_cost_formulas.py` |

Source: `src/flopscope/numpy/random/_cost_formulas.py`.

---

### Stats

Stats ops are composite (weight 1.0; all per-element factors in `flop_cost`).

| Op | flop_cost (per element) | basis |
|---|---|---|
| `stats.norm.ppf` | 83 | DERIVED composite: Acklam degree-5 rational + Newton step (erf + pdf + correction) + affine; measured 83.05 FP-instr/elem (empirical-weights.md); confirmed issue audit |
| `stats.norm.pdf` | ≈27 | DERIVED: exp + affine normalization |
| `stats.norm.cdf` | ≈48 | DERIVED: erf + affine |
| `stats.truncnorm.ppf` | 81 | DERIVED composite (affine + rational + Newton with erf+exp); calibration 82.52; confirmed issue audit |
| `stats.lognorm.ppf` | 106 | DERIVED composite (ndtri + exp); calibration 106.35; confirmed issue audit |
| `stats.lognorm.pdf` | 62 | DERIVED composite: log + exp + arithmetic per element; audit-2 verified; calibration alpha 62.30 |
| `stats.lognorm.cdf` | 70 | DERIVED composite: log + erf rational approx + arithmetic; audit-2 verified; calibration alpha 69.98 |
| `stats.laplace.pdf` | 1 | DECLARED: abs + exp per elem; transcendental tier (weight 16.0) |
| `stats.laplace.cdf` | 40 | DERIVED composite: two eager exp branches + 8 arith/cmp/select; audit-2 verified |
| `stats.laplace.ppf` | 51 | DERIVED composite: two eager log branches + edge selects; audit-2 verified |
| `stats.uniform.pdf` | 1 | DECLARED: 1 FLOP/elem |
| `stats.uniform.cdf` | 4 | DERIVED composite: sub + div + 2 clip compare/selects; calibrated alpha 4.0 |
| `stats.cauchy.pdf` | 6 | DERIVED pure-arithmetic: z=(x−loc)/scale; 1/(π·scale·(1+z²)) = 6 FLOPs/elem; weight 1.0; calibrated alpha 6.0 |

Source: `src/flopscope/stats/`.

---

### Window

| Op | flop_cost | basis | source |
|---|---|---|---|
| `bartlett` | `4n` (weight 1.0) | DERIVED: compare + divide + add + select per sample (FMA=2, 4 ops/sample) | `_window.py:bartlett_cost` |
| `blackman` | `40n` (weight 1.0) | DERIVED composite: 2 cosine evals at transcendental rate (16/elem each) + 8 mul/div/add per sample; all folded into flop_cost | `_window.py:blackman_cost` |
| `hamming` | `2n` (weight 8.0) | DECLARED: cosine eval per sample at the half-transcendental tier | `_window.py:hamming_cost` |
| `hanning` | `2n` (weight 8.0) | DECLARED: cosine eval per sample at the half-transcendental tier | `_window.py:hanning_cost` |
| `kaiser` | `23n` (weight 1.0) | DERIVED composite: 1 Bessel I₀ eval at transcendental tier (16/elem) + 7 scalar FLOPs per sample; folded into flop_cost | `_window.py:kaiser_cost` |

Source: `src/flopscope/_window.py`.

---

### Interp and histogram

| Op | flop_cost | basis | source |
|---|---|---|---|
| `interp` | `numel(xp) + m × ⌈log₂(numel(xp))⌉` (search + interpolate) | DERIVED | `_counting_ops.py` |
| `histogram` (integer bins) | `n × ⌈log₂(bins)⌉ + n` (binning pass + sort-based bin-edge search) | DERIVED | `_counting_ops.py` |
| `histogram` (string bins, e.g. `'auto'`) | `n × (2 + estimator_cost + ⌈log₂ resolved_bins⌉)` (deferred: resolved after the call; estimator costs: sturges/sqrt/rice=0, fd/auto=+1n, scott=+4n, doane=+6n, stone=+max(100,√n)n) | DERIVED | `_counting_ops.py` |
| `histogram2d`, `histogramdd` | same as `histogram` per axis | DERIVED | `_counting_ops.py` |
| `histogram_bin_edges` | `n × ⌈log₂ bins⌉` (integer bins) | DECLARED | `_counting_ops.py` |
| `trapezoid`, `trapz` | `4 × numel(y)` | DERIVED: `(d·(y₁+y₂)/2).sum()` ≈ 3 elementwise ops + sum-reduce per point, charged as a clean 4/point upper bound | `_pointwise.py`; fixed in this branch |

Source: `src/flopscope/_counting_ops.py`, `src/flopscope/_free_ops.py`.

---

### Set ops

| Op | flop_cost | basis |
|---|---|---|
| `unique`, `unique_all`, `unique_counts`, `unique_inverse`, `unique_values` | `n × ⌈log₂ n⌉` | DECLARED sort-based |
| `in1d`, `isin` | `(n+m) × ⌈log₂(n+m)⌉` | DECLARED sort-based |
| `intersect1d`, `setdiff1d`, `setxor1d`, `union1d` | `(n+m) × ⌈log₂(n+m)⌉` | DECLARED sort-based |
| `searchsorted` | `m × ⌈log₂ n⌉` | DECLARED binary search |

Comparison = 1 FLOP convention; weight 1.0.

---

### Counting (diff, ediff1d, clip, allclose, isclose, count_nonzero, trace)

| Op | flop_cost | basis | source |
|---|---|---|---|
| `clip` | `max(n_bounds, 1) × numel(output)` (1 compare-select per bound; n_bounds=0,1,2; floor of 1 ensures materialising copy is not free) | DERIVED | `_pointwise.py` |
| `count_nonzero` | `numel(input)` (every element tested regardless of axis; comparison-scan model) | DECLARED | `_pointwise.py` |
| `diff` | `prod(a.shape[:ax]) × (n×L − n×(n+1)/2) × prod(a.shape[ax+1:])`, `L = a.shape[ax]` | DERIVED: `n` passes of `L−k` subtractions | `_pointwise.py` |
| `ediff1d` | `ary.size − 1 + size(to_begin) + size(to_end)` | DECLARED | `_pointwise.py` |
| `gradient` | base: `sum_ax 2·S·(L−2)/L`; each coord-array axis adds a spacing surcharge (uniform: `+3(L−1)`; non-uniform: `+3S(L−2)/L + 10(L−2) + 3(L−1) + 4S/L`) | DERIVED | `_pointwise.py:gradient` |
| `allclose` | `7·numel(broadcast) − 1` (6 FLOPs/elem tolerance core + numel−1 all-reduce) | DERIVED | `_counting_ops.py` |
| `isclose` | `6·numel(broadcast)` (sub + 2·abs + mul + add + cmp per element) | DECLARED | `_pointwise.py` |
| `trace` (numpy.trace) | `min(ax1, ax2) × n_traces` where `n_traces = size / (shape[ax1] × shape[ax2])` (batch-multiplied) | DERIVED | `_counting_ops.py:trace` |
| `correlate` | mode-aware: `full` = `2nm−n−m+1`; `valid` = `(2·min−1)·(max−min+1)`; `same` = exact dot-length sum per numpy C layout | DERIVED per-mode | `_pointwise.py:_correlate_cost` |

---

### Copy and gather

Operations that materialize or scatter memory — no arithmetic, but billed
for the elements touched.  All use **weight 1.0** unless noted (weight 4.0
for gather-tier scatter ops).

| Op | flop_cost | basis | source |
|---|---|---|---|
| `insert` | `numel(output)` | DECLARED: np.insert allocates and copies arr + values | `_free_ops.py` |
| `append` | `numel(output)` = arr.size + values.size | DECLARED: np.append = concatenate | `_free_ops.py` |
| `delete` | `numel(output)` | DECLARED: surviving elements copied | `_free_ops.py` |
| `copyto` | elements written (numel(dst) when `where=True`; count_nonzero(broadcast(where)) otherwise) | DECLARED | `_free_ops.py` |
| `hstack` | `numel(output)` | DECLARED: allocates horizontally | `_free_ops.py` |
| `column_stack` | `numel(output)` | DECLARED: allocates as 2-D column array | `_free_ops.py` |
| `row_stack` | `numel(output)` (alias for vstack) | DECLARED | `_free_ops.py` |
| `tril`, `triu` | `numel(output)` | DECLARED: numpy returns a copy | `_free_ops.py` |
| `roll` | `numel(output)` | DECLARED: cyclic copy | `_free_ops.py` |
| `put` | `numel(indices)` (scatter writes; mode-independent) | DECLARED | `_free_ops.py` |
| `put_along_axis` | `(numel(arr) / arr.shape[axis]) × indices.shape[axis]`; `numel(indices)` when `axis=None` | DECLARED gather-tier (weight 4.0) | `_free_ops.py` |

---

### View / free (weight 0.0)

**Family rule**: operations that return a view, re-interpret memory, or
inspect metadata without touching element values charge 0 FLOPs.

Includes: `reshape`, `ravel`, `flatten`, `transpose`, `squeeze`,
`expand_dims`, `broadcast_to`, `atleast_1d/2d/3d`, `asarray` (no copy),
`asfortranarray`, `ascontiguousarray`, `astype` (no copy), `view`,
`diagonal` (view path), `diag` (view path), `squeeze`, `moveaxis`, `swapaxes`,
`ndim`, `shape`, `size`, `nbytes`, `itemsize`, `dtype`, `flags`, `base`,
`data`, `ctypes`, `strides`, `T`, `linalg.diagonal`, `linalg.matrix_transpose`,
`fft.fftfreq`, `fft.rfftfreq`, `fft.fftshift`, `fft.ifftshift`,
`isscalar`, `isfortran`, `ndim` attribute.

Source: `src/flopscope/_free_ops.py`.

---

## Evidence appendix: iterative linalg constants

The constants `eig=25n³`, `eigvals=10n³`, `eigh=9n³`, `eigvalsh=4n³/3`,
`svd-thin=6ab²+20b³`, `svd-full=4a²b+22b³`, `svdvals=2ab²+2b³` were
confirmed in June 2026 by three independent legs.

**Leg (a) — LAPACK driver op-counts**

| Op | LAPACK driver | Standard FLOP count |
|---|---|---|
| `cholesky` | dpotrf | n³/3 |
| `solve` | dgesv (= dgetrf + dgetrs) | 2n³/3 + 2n²/RHS |
| `inv` | dgetrf + dgetri | ≈2n³ |
| `det`, `slogdet` | dgetrf | 2n³/3 |
| `qr` (reduced) | dgeqrf + dorgqr | 2(2mn² − 2n³/3), k=min(m,n) |
| `eig` | dgeev (jobvr=V) | ≈25n³ (LAPACK Users' Guide Table 3.13 = 26.33) |
| `eigvals` | dgeev (jobvl=N, jobvr=N) | 10.00n³ (LUG Table 3.13, exact) |
| `eigh` | dsyevd | ≈9n³ |
| `eigvalsh` | dsyevd (jobz=N) | ≈4n³/3 |
| `svd` (thin) | dgesdd | 6ab² + 20b³ (G&VL 4e §8.6 Table 8.6.1) |
| `svd` (full, m≠n) | dgesdd | 4a²b + 22b³ |
| `svdvals` | dgesdd (jobz=N) | 2ab² + 2b³ |

References: LAPACK Users' Guide 3rd ed. Table 3.13; G&VL 4e §7.5 (eig), §8.3
(eigh/eigvalsh), §8.6 (SVD); LAWN 41 (QR).

**Leg (b) — Runtime scaling relative to Cholesky**

cholesky ≡ n³/3 FLOPs (dpotrf, anchor).  Implied constant for op X:
`implied_c = (t_X / t_cholesky) × (1/3)`.  See BLAS caveat below.

| Op | log-log slope | rel/chol @512 | rel/chol @768 | implied c @512 | implied c @768 | charged c | verdict |
|---|---|---|---|---|---|---|---|
| `eigvals` | 2.228 | 73.76 | 190.42 | 24.59 | 63.47 | 10.0 | **low** |
| `eig` | 2.135 | 118.31 | 216.33 | 39.44 | 72.11 | 25.0 | **low** |
| `eigvalsh` | 2.043 | 8.42 | 12.92 | 2.81 | 4.31 | 1.333 | **low** |
| `eigh` | 1.584 | 13.31 | 25.88 | 4.44 | 8.63 | 9.0 | **supports** |
| `svdvals` | 1.491 | 9.43 | 9.99 | 3.14 | 3.33 | 4.0 | **supports** |
| `svd` | 2.019 | 24.58 | 30.49 | 8.19 | 10.16 | 26.0 | **high** |
| `cholesky` | 1.594 | 1.00 | 1.00 | 0.333 | 0.333 | 0.333 | **supports** |
| `solve` | 2.072 | 1.30 | 0.97 | 0.433 | 0.324 | 0.671 | **supports** |
| `qr` | 1.579 | 4.83 | 7.08 | 1.61 | 2.36 | 2.667 | **supports** |
| `inv` | 1.505 | 2.22 | 2.98 | 0.739 | 0.992 | 2.0 | **high** |
| `det` | 1.544 | 1.27 | 0.90 | 0.424 | 0.299 | 0.667 | **supports** |

Raw timings (median of 5 runs, float64, `numpy.random.default_rng(42)`):

| Op | n=192 ms | n=256 ms | n=384 ms | n=512 ms | n=768 ms |
|---|---|---|---|---|---|
| `eigvals` | 43.6 | 92.9 | 232.2 | 421.1 | 978.4 |
| `eig` | 67.3 | 110.1 | 289.0 | 675.4 | 1111.5 |
| `eigvalsh` | 5.1 | 6.9 | 35.8 | 48.1 | 66.4 |
| `eigh` | 18.2 | 17.9 | 67.8 | 76.0 | 133.0 |
| `svdvals` | 9.0 | 9.8 | 32.6 | 53.9 | 51.3 |
| `svd` | 10.8 | 25.4 | 69.6 | 140.3 | 156.7 |
| `cholesky` | 0.7 | 1.3 | 3.4 | 5.7 | 5.1 |
| `solve` | 0.5 | 0.5 | 2.7 | 7.4 | 5.0 |
| `qr` | 3.1 | 13.7 | 12.3 | 27.6 | 36.4 |
| `inv` | 1.3 | 7.8 | 3.9 | 12.7 | 15.3 |
| `det` | 0.6 | 1.8 | 1.1 | 7.3 | 4.6 |

> **BLAS caveat**: wall-clock ratios are informative for compute-bound BLAS-3
> kernels but do NOT isolate n³ work alone — iteration counts vary per input,
> cache effects differ by n, and parallel thread counts may differ.  Treat
> `verdict_hint` as supporting signal (leg b of three), not a definitive count.

**Leg (c) — Textbook citations**

- `eig` 25n³: G&VL 4e §7.5 Hessenberg reduction (~10/3 n³) + Francis
  double-shift QR + eigenvector backtransform (~25n³ total); corroborated by
  LAPACK Users' Guide Table 3.13 DGEEV-with-vectors = 26.33n³.
- `eigvals` 10n³: LAPACK Users' Guide Table 3.13 DGEEV values-only = 10.00n³
  (exact entry).
- `eigh` 9n³: G&VL 4e §8.3 (DSYEVD: tridiagonalization + divide-and-conquer
  with eigenvectors).
- `eigvalsh` 4n³/3: G&VL 4e §8.3 (DSYEVD tridiagonalization only, MRRR, no
  eigenvectors).
- `svd` thin 6ab²+20b³, full 4a²b+22b³: G&VL 4e §8.6 Table 8.6.1 (R-SVD).
- `svdvals` 2ab²+2b³: G&VL 4e §8.6 Table 8.6.1 Σ-only row.

**Per-op verdict summary**

| Op | charged constant | leg-b verdict | leg-c | overall |
|---|---|---|---|---|
| `eig` | 25n³ | low (implied ~39–72n³) | G&VL/LUG 25–26n³ | keep (textbook-anchored) |
| `eigvals` | 10n³ | low (implied ~25–63n³) | LUG exact 10n³ | keep (exact citation) |
| `eigh` | 9n³ | supports (implied ~4–9n³) | G&VL 9n³ | keep |
| `eigvalsh` | 4n³/3 | low (implied ~3–4n³) | G&VL 4n³/3 | keep |
| `svd` (thin) | 6ab²+20b³ | high (implied ~8–10n³ vs 26n³ @sq) | G&VL Table 8.6.1 | keep |
| `svd` (full) | 4a²b+22b³ | — | G&VL Table 8.6.1 | keep |
| `svdvals` | 2ab²+2b³ | supports | G&VL Table 8.6.1 | keep |
| `cholesky` | n³/3 | supports | G&VL Alg 4.2.1 | keep |
| `solve` | 2n³/3+2n²/rhs | supports | G&VL §3.2 / LUG 0.67n³ | keep |
| `qr` | 2(2mn²−2n³/3) | supports | G&VL §5.2 / LAWN 41 | keep |
| `inv` | 2n³ | high (implied ~0.7–1.0n³) | G&VL §3.4 | overcharges; retained |
| `det` | 2n³/3 | supports | G&VL §3.2 LU only | keep |

