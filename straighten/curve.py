from typing import Union, Sequence, Callable

import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.interpolate import interp1d, UnivariateSpline

ShapeLike = Union[int, Sequence[int]]


def cumulative_length(curve: np.ndarray):
    lengths = np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))
    lengths = np.insert(lengths, 0, 0)
    return lengths


def get_derivatives(curve: np.ndarray, step: float):
    assert curve.ndim == 2
    _, d = curve.shape

    lengths = cumulative_length(curve)
    xs = np.arange(0, lengths[-1], step)
    yield interp1d(lengths, curve, axis=0)(xs)

    grad = curve
    for _ in range(d):
        grad = np.gradient(grad, axis=0)
        yield interp1d(lengths, grad, axis=0)(xs)


def frenet_serret(*gradients):
    _, d = gradients[0].shape
    basis = []
    for grad in gradients:
        e = grad
        # Gram-Schmidt process
        for v in basis:
            e = e - v * (v * grad).sum(axis=-1, keepdims=True)

        e /= np.linalg.norm(e, axis=-1, keepdims=True)
        basis.append(e)

    return np.stack(basis, -1)


def simplify(curve: np.ndarray, order: int = 3, smoothing=100, n_points=100, extrapolate=(0, 0)):
    # natural parametrization
    lengths = cumulative_length(curve)

    extrapolate = np.broadcast_to(extrapolate, 2)
    alpha = np.linspace(-extrapolate[0], lengths[-1] + extrapolate[1], n_points)
    # fit a spline for each dimension
    return np.stack([
        UnivariateSpline(lengths, coords, k=order, s=smoothing, ext='extrapolate')(alpha) for coords in curve.T
    ], 1)


def get_local_coordinates(curve):
    result = np.zeros_like(curve)
    result[:, 0] = cumulative_length(curve)
    return result


def switch_basis(point, source_origins, target_origins, target_basis):
    points = point - source_origins
    idx = np.linalg.norm(points, axis=-1).argmin()
    local = np.einsum('nij,nj->ni', target_basis, points)
    distances = local[:, 0]

    # how many good planes are there?
    candidates, = np.diff(np.sign(distances)).nonzero()
    # choose the closest one to the basis' origin
    idx = candidates[np.abs(candidates - idx).argmin()]
    slc = slice(max(0, idx - 2), idx + 2)

    distances = distances[slc]
    local = local[slc] + target_origins[slc]
    # ensure that there is exactly one zero
    assert len(np.diff(np.sign(distances)).nonzero()[0]) == 1
    return interp1d(distances, local, axis=0)(0)


class Interpolator:
    def __init__(self, curve: np.ndarray, step: float, spacing: ShapeLike = 1,
                 get_local_basis: Callable = frenet_serret):
        """
        Parameters
        ----------
        curve: array of shape (n_points, dim)
        step: step size along the curve
        get_local_basis: Callable(*gradients) -> local_basis
        spacing: the nd-pixel spacing

        Returns
        -------
        grid: (dim, n_points, *shape)
        """
        assert curve.ndim == 2
        self.dim = curve.shape[1]
        self.step = step
        self.spacing = np.broadcast_to(spacing, self.dim)
        self.get_local_basis = get_local_basis

        self.curve = curve * spacing
        self.even_curve, *grads = get_derivatives(self.curve, step)
        self.basis = get_local_basis(*grads)

    def get_grid(self, shape: ShapeLike):
        shape = np.broadcast_to(shape, self.dim - 1)
        grid = np.meshgrid(*(np.arange(s) - s / 2 for s in shape))
        zs = np.zeros_like(grid[0])
        grid = np.stack([zs, *grid])

        grid = np.einsum('Nij,j...->Ni...', self.basis, grid)
        grid = np.moveaxis(grid, [0, 1], [-2, -1])
        grid = (grid + self.even_curve) / self.spacing
        return np.moveaxis(grid, [-2, -1], [1, 0])

    def interpolate_along(self, array, shape, fill_value=0, order=1):
        """
        shape: the desired shape of the planes
        """
        if callable(fill_value):
            fill_value = fill_value(array)
        return map_coordinates(array, self.get_grid(shape), order=order, cval=fill_value)

    def _get_centers(self, shape):
        centers = np.zeros_like(self.even_curve)
        centers[:, 0] = cumulative_length(self.even_curve)
        centers[:, 1:] = shape / 2
        return centers

    def _to_local(self, point, shape):
        return switch_basis(point, self.even_curve, self._get_centers(shape), np.moveaxis(self.basis, -1, -2))

    def _to_global(self, point, shape):
        return switch_basis(point, self._get_centers(shape), self.even_curve, self.basis)

    @staticmethod
    def _transform(points, shape, func):
        # point: *any, dim
        *spatial, d = points.shape
        shape = np.broadcast_to(shape, d - 1)
        points = points.reshape(-1, d)
        results = []
        for p in points:
            results.append(func(p, shape))

        return np.array(results).reshape(*spatial, d)

    def _check_points(self, points):
        points = np.asarray(points)
        *spatial, d = points.shape
        assert d == self.dim
        return points

    def global_to_local(self, points, shape):
        return self._transform(self._check_points(points) * self.spacing, shape, self._to_local)

    def local_to_global(self, points, shape):
        return self._transform(self._check_points(points), shape, self._to_global) / self.spacing
