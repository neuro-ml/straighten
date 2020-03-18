import warnings
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


def interpolate_coords(coordinates, distance_to_origin, distance_to_plane):
    idx = distance_to_origin.argmin()

    # how many good planes are there?
    candidates, = np.diff(np.sign(distance_to_plane)).nonzero()
    # choose the closest one to the basis' origin
    idx = candidates[np.abs(candidates - idx).argmin()]
    slc = slice(max(0, idx - 2), idx + 2)

    distance_to_plane = distance_to_plane[slc]
    coordinates = coordinates[slc]
    # ensure that there is exactly one zero
    if len(np.diff(np.sign(distance_to_plane)).nonzero()[0]) != 1:
        warnings.warn("Couldn't choose a local basis.")
    return interp1d(distance_to_plane, coordinates, axis=0)(0)


def identity(x):
    return x


class Interpolator:
    def __init__(self, curve: np.ndarray, step: float, spacing: ShapeLike = 1,
                 get_local_basis: Callable = frenet_serret, shift_basis: Callable = identity):
        """
        Parameters
        ----------
        curve: array of shape (n_points, dim)
        step: step size along the curve
        spacing: the nd-pixel spacing
        get_local_basis: Callable(curve) -> local_basis

        Returns
        -------
        grid: (dim, n_points, *shape)
        """
        assert curve.ndim == 2
        self.dim = curve.shape[1]
        self.spacing = np.broadcast_to(spacing, self.dim)

        even_curve, *grads = get_derivatives(curve * spacing, step)
        self.even_curve = shift_basis(even_curve)
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
        points = point - self.even_curve
        to_origin = np.linalg.norm(points, axis=-1)

        points = np.einsum('nji,nj->ni', self.basis, points)
        to_plane = points[:, 0]

        return interpolate_coords(points + self._get_centers(shape), to_origin, to_plane)

    def _to_global(self, point, shape):
        points = point - self._get_centers(shape)
        to_plane = points[:, 0]

        points = np.einsum('nij,nj->ni', self.basis, points)
        to_origin = np.linalg.norm(points, axis=-1)

        return interpolate_coords(points + self.even_curve, to_origin, to_plane)

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
        assert d == self.dim, (d, self.dim)
        return points

    def global_to_local(self, points, shape):
        return self._transform(self._check_points(points) * self.spacing, shape, self._to_local)

    def local_to_global(self, points, shape):
        return self._transform(self._check_points(points), shape, self._to_global) / self.spacing
