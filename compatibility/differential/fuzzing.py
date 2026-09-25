"""Seeded, declarative input fuzzing for the differential suites.

WHY: the hand-written differential tests pick their shapes by hand, and a
hand-picked shape is almost always a fresh C-contiguous array. The cases
where a fast path is wrong or loses - strided and reversed views,
transposed layouts, odd ranks, empty arrays, NaN and inf sprinkled in -
are exactly the ones nobody writes down. This module describes an INPUT
SPACE instead of an input, samples it reproducibly from a seed, and, when a
sample makes a patched call diverge from stock NumPy, shrinks that sample
and stores it as a permanent corpus case that every plain test run replays.

Two upstream ideas, reimplemented for ndarrays (ideas only, no code copied):
- PyTorch's benchmark fuzzer (torch/utils/benchmark/utils/fuzzer.py):
  named size symbols drawn from distributions, a probability of
  contiguity, stepped slicing for strided views, element and allocation
  limits, variable rank, and resampling until every constraint holds.
- Go's native fuzzing (src/internal/fuzz): a failing input is minimized,
  then written under a per-target corpus directory named by its content
  hash, and from then on runs as an ordinary test case.

Layout is part of a case: a corpus file stores the logical values plus the
recipe (axis order and per-axis step) that rebuilds the exact view, because
saving the view itself would silently make it contiguous again.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX, StockRaised

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"


# --- the input space ---------------------------------------------------------


@dataclass(frozen=True)
class FuzzedParameter:
    """One named scalar drawn per sample.

    ``distribution`` is "uniform" or "loguniform" over the integers
    [minval, maxval], or a {value: probability} dict for a discrete choice
    (values may be None, meaning "leave the argument out").
    """

    name: str
    minval: int = 0
    maxval: int = 0
    distribution: str | dict = "uniform"

    def sample(self, rng: np.random.Generator):
        dist = self.distribution
        if isinstance(dist, dict):
            values = list(dist)
            probs = np.asarray(list(dist.values()), dtype=np.float64)
            return values[int(rng.choice(len(values), p=probs / probs.sum()))]
        if dist == "uniform":
            return int(rng.integers(self.minval, self.maxval, endpoint=True))
        if dist == "loguniform":
            # drawn over [minval + 1, maxval + 2) and shifted back, so a
            # minval of 0 (an empty axis) stays reachable
            lo, hi = math.log(self.minval + 1), math.log(self.maxval + 2)
            return min(int(math.exp(rng.uniform(lo, hi))) - 1, self.maxval)
        raise ValueError(f"unknown distribution {dist!r} for {self.name}")


@dataclass(frozen=True)
class Layout:
    """How a logical array sits in memory: ``perm`` lists the axes from the
    outermost stored axis inward, ``step`` is the per-axis slicing step
    (negative = reversed view). Identity perm and all-ones step = C order."""

    perm: tuple[int, ...]
    step: tuple[int, ...]

    @staticmethod
    def contiguous(ndim: int) -> "Layout":
        return Layout(tuple(range(ndim)), (1,) * ndim)

    @property
    def is_contiguous(self) -> bool:
        return self == Layout.contiguous(len(self.perm))


def materialize(values: np.ndarray, layout: Layout) -> np.ndarray:
    """A fresh view holding ``values`` with ``layout``'s strides."""
    if values.ndim == 0:
        return values.copy()
    perm, step = layout.perm, layout.step
    storage = np.zeros(
        tuple(values.shape[p] * abs(step[p]) for p in perm), dtype=values.dtype
    )
    view = storage[tuple(slice(None, None, step[p]) for p in perm)]
    view[...] = values.transpose(perm)
    return view.transpose(np.argsort(perm))


def _storage_bytes(values: np.ndarray, layout: Layout) -> int:
    n = values.dtype.itemsize
    for axis, length in enumerate(values.shape):
        n *= length * abs(layout.step[axis])
    return n


_SPECIALS = (np.nan, np.inf, -np.inf, -0.0)


def default_values(
    rng: np.random.Generator, dtype: np.dtype, shape: tuple, special_probability: float
) -> np.ndarray:
    """Values that stress comparisons: wide magnitudes, ties, dtype
    extremes, and (floats) NaN / +-inf / -0.0 sprinkled in."""
    kind = dtype.kind
    if kind == "b":
        return rng.random(shape) < 0.5
    if kind in "iu":
        info = np.iinfo(dtype)
        if rng.random() < 0.3:  # heavy ties
            lo, hi = max(info.min, -3), min(info.max, 3)
        else:
            lo, hi = info.min, info.max
        return rng.integers(lo, hi, size=shape, dtype=dtype, endpoint=True)
    scale = 10.0 ** rng.uniform(-3, 6)
    vals = np.asarray(rng.standard_normal(shape)) * scale
    if rng.random() < 0.2:
        vals = np.round(vals / scale * 2)  # ties among floats
    if kind == "c":
        vals = vals + 1j * (np.asarray(rng.standard_normal(shape)) * scale)
    vals = vals.astype(dtype)
    if vals.size and rng.random() < special_probability:
        mask = rng.random(shape) < rng.uniform(0.0, 0.2)
        picks = rng.choice(len(_SPECIALS), size=int(mask.sum()))
        vals[mask] = np.asarray(_SPECIALS, dtype=np.float64)[picks]
    return vals


@dataclass(frozen=True)
class FuzzedArray:
    """One named ndarray drawn per sample.

    ``size`` holds, per axis, an int or the name of a FuzzedParameter;
    ``rank`` (a parameter name) keeps only the first that many axes, so one
    spec covers several ranks. With probability 1 - probability_contiguous
    the array comes back as a non-contiguous view (a random axis order
    and/or stepped slicing, possibly reversed).
    """

    name: str
    size: tuple
    dtypes: tuple = (np.float64,)
    rank: str | None = None
    probability_contiguous: float = 0.5
    steps: tuple[int, ...] = (1, 2, 3, -1, -2)
    min_elements: int = 0
    max_elements: int = 1 << 16
    max_allocation_bytes: int = 1 << 22
    special_probability: float = 0.3
    values: Callable = default_values

    def sample(self, rng: np.random.Generator, params: dict) -> tuple[np.ndarray, Layout]:
        dtype = np.dtype(self.dtypes[int(rng.integers(len(self.dtypes)))])
        ndim = params[self.rank] if self.rank is not None else len(self.size)
        shape = tuple(
            params[s] if isinstance(s, str) else int(s) for s in self.size[:ndim]
        )
        values = np.asarray(self.values(rng, dtype, shape, self.special_probability))
        if ndim == 0 or rng.random() < self.probability_contiguous:
            return values, Layout.contiguous(ndim)
        perm = tuple(int(p) for p in rng.permutation(ndim))
        step = [int(self.steps[int(rng.integers(len(self.steps)))]) for _ in range(ndim)]
        if perm == tuple(range(ndim)) and all(s == 1 for s in step):
            step[int(rng.integers(ndim))] = 2  # asked for non-contiguous: make it so
        return values, Layout(perm, tuple(step))

    def admits(self, values: np.ndarray, layout: Layout) -> bool:
        return (
            self.min_elements <= values.size <= self.max_elements
            and _storage_bytes(values, layout) <= self.max_allocation_bytes
        )


@dataclass
class Case:
    """One concrete input: logical values + layout per array, plus scalars."""

    arrays: dict[str, tuple[np.ndarray, Layout]]
    params: dict = field(default_factory=dict)
    origin: dict = field(default_factory=dict)  # seed/index that produced it

    def materialize(self) -> dict[str, np.ndarray]:
        return {k: materialize(v, lay) for k, (v, lay) in self.arrays.items()}


@dataclass(frozen=True)
class Fuzzer:
    """Samples Cases from parameters + arrays, resampling until every
    array admits its draw and every ``constraints`` callable (given the
    params dict) holds."""

    parameters: tuple[FuzzedParameter, ...]
    arrays: tuple[FuzzedArray, ...]
    constraints: tuple[Callable[[dict], bool], ...] = ()
    max_attempts: int = 1000

    def take(self, n: int, seed: int) -> Iterator[Case]:
        rng = np.random.default_rng(seed)
        for index in range(n):
            yield self._one(rng, {"seed": seed, "index": index})

    def _one(self, rng: np.random.Generator, origin: dict) -> Case:
        for _ in range(self.max_attempts):
            params = {p.name: p.sample(rng) for p in self.parameters}
            if not all(c(params) for c in self.constraints):
                continue
            drawn = {}
            for spec in self.arrays:
                values, layout = spec.sample(rng, params)
                if not spec.admits(values, layout):
                    break
                drawn[spec.name] = (values, layout)
            else:
                return Case(drawn, params, origin)
        raise RuntimeError(
            f"no admissible sample in {self.max_attempts} attempts; "
            "the constraints leave the input space (nearly) empty"
        )


# --- one fast path's differential contract ------------------------------------


def exact(g: np.ndarray, e: np.ndarray) -> bool:
    return bool(np.array_equal(g, e, equal_nan=g.dtype.kind in "fc"))


def byte_exact(g: np.ndarray, e: np.ndarray) -> bool:
    """Raw-buffer equality: also tells +0.0 from -0.0 (caller-supplied
    fill values must keep their sign)."""
    return g.dtype == e.dtype and g.shape == e.shape and g.tobytes() == e.tobytes()


@dataclass(frozen=True)
class FuzzSpec:
    """The seeded input space for one fast path and how to call its op."""

    path: str
    op: str
    fuzzer: Fuzzer
    call: Callable[[dict, dict], tuple[tuple, dict]]
    equal: Callable[[np.ndarray, np.ndarray], bool] = exact
    enable_paths: tuple[str, ...] = ()  # paths registered disabled by default


class activated:
    """Patch exactly the spec's op (and switch on default-off paths) for
    the duration; restores stock NumPy on exit."""

    def __init__(self, spec: FuzzSpec):
        self.spec = spec

    def __enter__(self):
        pyoverdrive.enable([self.spec.op])
        for name in self.spec.enable_paths:
            pyoverdrive.enable_path(name)
        return self

    def __exit__(self, *exc):
        for name in self.spec.enable_paths:
            pyoverdrive.disable_path(name)
        pyoverdrive.disable()
        return False


@dataclass
class Outcome:
    dispatched: bool
    mismatch: str | None


def _public(op: str):
    holder = np
    parts = op.split(".")
    for p in parts[1:-1]:
        holder = getattr(holder, p)
    return getattr(holder, parts[-1])


def _path_object(op: str, name: str):
    for path in GEARBOX._paths.get(op, ()):
        if path.name == name:
            return path
    raise KeyError(f"{op} has no fast path named {name!r}")


def _invoke(fn, args, kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return fn(*args, **kwargs), None
        except StockRaised as sr:  # a path surfacing stock's own exception
            return None, sr.original
        except Exception as exc:  # noqa: BLE001 - parity is the contract
            return None, exc


def _differ(expected, got, equal) -> str | None:
    if isinstance(expected, tuple) or isinstance(got, tuple):
        if not (isinstance(expected, tuple) and isinstance(got, tuple)):
            return f"tuple-ness {type(got).__name__} != {type(expected).__name__}"
        if len(got) != len(expected):
            return f"tuple length {len(got)} != {len(expected)}"
        for i, (g, e) in enumerate(zip(got, expected)):
            why = _differ(e, g, equal)
            if why:
                return f"element {i}: {why}"
        return None
    if type(got) is not type(expected):
        return f"type {type(got).__name__} != {type(expected).__name__}"
    ge, ee = np.asarray(got), np.asarray(expected)
    if ge.shape != ee.shape:
        return f"shape {ge.shape} != {ee.shape}"
    if ge.dtype != ee.dtype:
        return f"dtype {ge.dtype} != {ee.dtype}"
    if not equal(ge, ee):
        return "values differ"
    return None


def check_case(spec: FuzzSpec, case: Case) -> Outcome:
    """Patched public name vs stock, each on its own fresh materialization
    (so an input-mutating path cannot hide behind a shared buffer). When
    the path dispatches, its run is also called directly: a run that raises
    would otherwise be masked by the Gearbox's once-per-path fallback."""
    args, kwargs = spec.call(case.materialize(), case.params)
    decision, _ = GEARBOX.decide(spec.op, args, kwargs)
    dispatched = decision == spec.path
    expected, stock_exc = _invoke(GEARBOX.stock_fn(spec.op), *spec.call(case.materialize(), case.params))
    got, got_exc = _invoke(_public(spec.op), args, kwargs)
    if stock_exc is not None or got_exc is not None:
        if type(got_exc) is not type(stock_exc):
            return Outcome(dispatched, f"exception: stock {stock_exc!r} vs patched {got_exc!r}")
    else:
        why = _differ(expected, got, spec.equal)
        if why:
            return Outcome(dispatched, why)
    if dispatched:
        _, run_exc = _invoke(_path_object(spec.op, spec.path).run, *spec.call(case.materialize(), case.params))
        if run_exc is not None and type(run_exc) is not type(stock_exc):
            return Outcome(dispatched, f"fast path run raised {run_exc!r} (users get a fallback warning)")
    return Outcome(dispatched, None)


# --- shrinking (after Go's minimizer: drop chunks, then simplify) --------------


def _with_array(case: Case, name: str, values: np.ndarray, layout: Layout) -> Case:
    arrays = dict(case.arrays)
    arrays[name] = (values, layout)
    return replace(case, arrays=arrays)


def _candidates(case: Case, name: str) -> Iterator[Case]:
    """Strictly simpler variants of one array, biggest reductions first:
    every candidate lowers (size, layout complexity, nonzeros, fractional
    values), so greedy acceptance always terminates."""
    values, layout = case.arrays[name]
    for axis, length in enumerate(values.shape):
        if length < 2:
            continue
        half = length // 2
        cuts = [slice(0, half), slice(half, length)]
        if length <= 16:
            cuts += [np.delete(np.arange(length), i) for i in range(length)]
        for cut in cuts:
            index = [slice(None)] * values.ndim
            index[axis] = cut
            yield _with_array(case, name, np.ascontiguousarray(values[tuple(index)]), layout)
    if not layout.is_contiguous:
        yield _with_array(case, name, values, Layout.contiguous(values.ndim))
        if any(s != 1 for s in layout.step):
            yield _with_array(case, name, values, Layout(layout.perm, (1,) * values.ndim))
        if layout.perm != tuple(range(values.ndim)):
            yield _with_array(case, name, values, Layout(tuple(range(values.ndim)), layout.step))
    flat = values.reshape(-1)
    if flat.size and values.dtype.kind != "b":
        half = flat.size // 2
        for chunk in (slice(0, half), slice(half, flat.size)):
            if np.count_nonzero(flat[chunk]):
                zeroed = flat.copy()
                zeroed[chunk] = 0
                yield _with_array(case, name, zeroed.reshape(values.shape), layout)
        if values.dtype.kind in "fc":
            with np.errstate(invalid="ignore"):
                rounded = np.round(values)
            if not np.array_equal(rounded, values, equal_nan=True):
                yield _with_array(case, name, rounded, layout)


def shrink(case: Case, fails: Callable[[Case], bool], budget: int = 400) -> Case:
    """Greedy minimization: accept the first still-failing candidate,
    restart, stop when nothing simpler fails or the budget is spent."""
    progressed = True
    while progressed and budget > 0:
        progressed = False
        for name in list(case.arrays):
            for candidate in _candidates(case, name):
                budget -= 1
                if fails(candidate):
                    case, progressed = candidate, True
                    break
                if budget <= 0:
                    return case
            if progressed:
                break
    return case


# --- the corpus -----------------------------------------------------------------


def case_digest(case: Case) -> str:
    h = hashlib.sha256()
    for name in sorted(case.arrays):
        values, layout = case.arrays[name]
        h.update(repr((name, values.dtype.str, values.shape, layout)).encode())
        h.update(np.ascontiguousarray(values).tobytes())
    h.update(json.dumps(case.params, sort_keys=True).encode())
    return h.hexdigest()[:16]


def save_case(spec: FuzzSpec, case: Case, mismatch: str, corpus: Path = CORPUS_DIR) -> Path:
    """Write <corpus>/<path>/<hash>.npz; the same case always gets the same
    name, so re-finding a known mismatch never duplicates it."""
    payload = {}
    for name, (values, layout) in case.arrays.items():
        payload[f"array:{name}"] = np.ascontiguousarray(values)
        payload[f"layout:{name}"] = np.array([layout.perm, layout.step], dtype=np.int64).reshape(2, -1)
    meta = {"path": spec.path, "op": spec.op, "params": case.params,
            "origin": case.origin, "mismatch": mismatch}
    payload["meta"] = np.array(json.dumps(meta, sort_keys=True))
    target = corpus / spec.path / f"{case_digest(case)}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez(buf, **payload)
    target.write_bytes(buf.getvalue())
    return target


def load_case(file: Path) -> tuple[dict, Case]:
    with np.load(file, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        arrays = {}
        for key in data.files:
            if key.startswith("array:"):
                name = key[len("array:"):]
                perm, step = data[f"layout:{name}"]
                arrays[name] = (
                    data[key],
                    Layout(tuple(int(p) for p in perm), tuple(int(s) for s in step)),
                )
    return meta, Case(arrays, meta["params"], meta.get("origin", {}))


def corpus_files(corpus: Path = CORPUS_DIR) -> list[Path]:
    return sorted(corpus.glob("*/*.npz"))


# --- a fuzzing session ------------------------------------------------------------


@dataclass
class Report:
    path: str
    samples: int = 0
    dispatched: int = 0
    mismatches: list[tuple[Case, str]] = field(default_factory=list)


def run_spec(spec: FuzzSpec, iterations: int, seed: int, shrink_budget: int = 400) -> Report:
    """Sample, check, and shrink every mismatch. Call inside activated(spec)."""
    report = Report(spec.path)
    for case in spec.fuzzer.take(iterations, seed):
        outcome = check_case(spec, case)
        report.samples += 1
        report.dispatched += outcome.dispatched
        if outcome.mismatch is not None:
            small = shrink(case, lambda c: check_case(spec, c).mismatch is not None, shrink_budget)
            report.mismatches.append((small, check_case(spec, small).mismatch or outcome.mismatch))
    return report


def describe(case: Case) -> str:
    parts = []
    for name, (values, layout) in case.arrays.items():
        shown = np.array2string(values, threshold=12, edgeitems=3)
        parts.append(f"{name}: {values.dtype} {values.shape} {layout} = {shown}")
    parts.append(f"params: {case.params}  origin: {case.origin}")
    return "; ".join(parts)
