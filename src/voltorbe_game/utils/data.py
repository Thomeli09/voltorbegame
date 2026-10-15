"""General-purpose operations on lists, vectors, and matrices.

Functions accept ordinary Python lists (including nested lists) and/or
NumPy arrays where appropriate. Inputs are not modified in place.

Selection functions return NumPy arrays for values and indices. For a
one-dimensional input, indices are integers; for multidimensional input,
indices contain coordinates in row-major order.

Examples
--------
sort_data([3, 1, 2], ["c", "a", "b"])
([1, 2, 3], ['a', 'b', 'c'])
multiply_list(2, [[1, 2], [3, 4]])
[[2, 4], [6, 8]]
sum_lists([[1, 2], [3, 4]])
[4, 6]
"""

from math import gcd
from typing import Any, overload

import numpy as np
from numpy.typing import NDArray


# Python 3.12+ generic type aliases (supported by Python 3.15).
type DataSequence[T] = list[T] | NDArray[Any]
type NestedNumbers = int | float | complex | np.number[Any] | list[NestedNumbers]
type NumericInput = list[NestedNumbers] | NDArray[Any]
type Position = int | tuple[int, ...]


def _validate_sequence(data: object, name: str) -> None:
    """Check that an input is a list or 1D/2D NumPy array."""
    if not isinstance(data, (list, np.ndarray)):
        raise TypeError(f"'{name}' must be a list or NumPy array.")
    if isinstance(data, np.ndarray) and data.ndim not in (1, 2):
        raise ValueError(f"'{name}' must be one- or two-dimensional.")


def _sort_key(item: Any) -> Any:
    """Use lexicographic comparisons for NumPy matrix rows."""
    return item.tolist() if isinstance(item, np.ndarray) else item


def _sorted_indices(primary: list[Any] | NDArray[Any]) -> list[int]:
    """Return indices of a stable, ascending, row-wise sort."""
    return sorted(
        range(len(primary)),
        key=lambda index: _sort_key(primary[index]),
    )


@overload
def sort_data[T](primary: list[T], secondary: None = None) -> list[T]: ...


@overload
def sort_data[T, U](
    primary: list[T], secondary: list[U]
) -> tuple[list[T], list[U]]: ...


@overload
def sort_data[T](
    primary: list[T], secondary: NDArray[Any]
) -> tuple[list[T], NDArray[Any]]: ...


@overload
def sort_data(primary: NDArray[Any], secondary: None = None) -> NDArray[Any]: ...


@overload
def sort_data[T](
    primary: NDArray[Any], secondary: list[T]
) -> tuple[NDArray[Any], list[T]]: ...


@overload
def sort_data(
    primary: NDArray[Any], secondary: NDArray[Any]
) -> tuple[NDArray[Any], NDArray[Any]]: ...


def sort_data[T, U](
    primary: DataSequence[T],
    secondary: DataSequence[U] | None = None,
) -> DataSequence[T] | tuple[DataSequence[T], DataSequence[U]]:
    """Sort a list/vector/matrix and optionally reorder associated data.

    Sorting is stable and ascending. A nested list or two-dimensional array
    is sorted **by entire rows**, lexicographically (not element by element).
    Both lists and arrays preserve their container type. ``secondary`` may
    be a vector or a matrix, and may use a different container type.

    Parameters
    ----------
    primary : list or numpy.ndarray
        One-dimensional keys, or rows to sort lexicographically.
    secondary : list or numpy.ndarray, optional
        Associated values or rows. Its first dimension must match the
        length of ``primary``.

    Returns
    -------
    list or numpy.ndarray or tuple
        Sorted primary data, or the sorted primary and associated data.

    Raises
    ------
    TypeError
        If an input is not a list or NumPy array, or its elements cannot
        be ordered.
    ValueError
        If array dimensions are unsupported or input lengths differ.

    Examples
    --------
    >>> sort_data([[2, 1], [1, 5], [2, 0]])
    [[1, 5], [2, 0], [2, 1]]
    >>> sort_data([3, 1, 2], [[30, 31], [10, 11], [20, 21]])
    ([1, 2, 3], [[10, 11], [20, 21], [30, 31]])
    """
    _validate_sequence(primary, "primary")
    if secondary is not None:
        _validate_sequence(secondary, "secondary")
        if len(primary) != len(secondary):
            raise ValueError("'primary' and 'secondary' must have equal lengths.")

    indices = _sorted_indices(primary)
    ordered_primary = (
        primary[indices] if isinstance(primary, np.ndarray)
        else [primary[index] for index in indices]
    )
    if secondary is None:
        return ordered_primary

    ordered_secondary = (
        secondary[indices] if isinstance(secondary, np.ndarray)
        else [secondary[index] for index in indices]
    )
    return ordered_primary, ordered_secondary


@overload
def find_first_max_pair[T, U](
    primary: list[T], secondary: list[U]
) -> tuple[T, U]: ...


@overload
def find_first_max_pair[T](
    primary: list[T], secondary: NDArray[Any]
) -> tuple[T, Any]: ...


@overload
def find_first_max_pair[U](
    primary: NDArray[Any], secondary: list[U]
) -> tuple[np.generic | NDArray[Any], U]: ...


@overload
def find_first_max_pair(
    primary: NDArray[Any], secondary: NDArray[Any]
) -> tuple[np.generic | NDArray[Any], np.generic | NDArray[Any]]: ...


def find_first_max_pair[T, U](
    primary: DataSequence[T],
    secondary: DataSequence[U],
) -> tuple[T | np.generic | NDArray[Any], U | np.generic | NDArray[Any]]:
    """Find the first maximum key and its associated value or row.

    When ``primary`` is a matrix, its maximum row is determined
    lexicographically. Ties always select the first occurrence.

    Parameters
    ----------
    primary : list or numpy.ndarray
        Non-empty keys or rows from which to find the maximum.
    secondary : list or numpy.ndarray
        Paired values or rows, of the same first-dimension length.

    Returns
    -------
    tuple
        Maximum value/row and its paired value/row. Returns array rows
        when given NumPy matrices.

    Raises
    ------
    TypeError
        If an input has an unsupported type or keys cannot be compared.
    ValueError
        If an input is empty, has invalid dimensions, or lengths differ.

    Examples
    --------
    >>> find_first_max_pair([3, 8, 8], ["a", "b", "c"])
    (8, 'b')
    """
    _validate_sequence(primary, "primary")
    _validate_sequence(secondary, "secondary")
    if len(primary) == 0 or len(secondary) == 0:
        raise ValueError("Inputs must not be empty.")
    if len(primary) != len(secondary):
        raise ValueError("'primary' and 'secondary' must have equal lengths.")

    index = max(range(len(primary)), key=lambda i: _sort_key(primary[i]))
    return primary[index], secondary[index]


def multiply_list(
    value: int | float | complex,
    values: NumericInput,
) -> NumericInput:
    """Multiply each numeric element in a list or array by a scalar.

    Python lists can be nested to any depth, including irregular shapes;
    NumPy vectors/matrices retain their original shape. Neither input
    is modified.

    Parameters
    ----------
    value : int, float or complex
        Multiplication factor.
    values : list or numpy.ndarray
        Numeric list (possibly nested) or numeric NumPy array.

    Returns
    -------
    list or numpy.ndarray
        New data with each numeric leaf multiplied by ``value``.

    Raises
    ------
    TypeError
        If scalar multiplication is unsupported for an element.

    Examples
    --------
    >>> multiply_list(3, [[1, 2], [3]])
    [[3, 6], [9]]
    """
    if isinstance(values, np.ndarray):
        return values * value
    if not isinstance(values, list):
        raise TypeError("'values' must be a list or NumPy array.")

    def multiply_nested(items: list[Any]) -> list[Any]:
        return [multiply_nested(item) if isinstance(item, list) else item * value
                for item in items]

    return multiply_nested(values)


def sum_lists(
    lists: list[NestedNumbers] | NDArray[Any],
) -> NestedNumbers | NDArray[Any] | np.generic:
    """Sum aligned values along the outermost dimension.

    For nested Python lists, all corresponding sublists must have the same
    lengths and nesting. With a NumPy array, this is ``np.sum(axis=0)``.
    A one-dimensional input yields a scalar sum; empty Python input yields
    an empty list. Nested lists retain their nested-list representation.

    Parameters
    ----------
    lists : list or numpy.ndarray
        Numeric list, list of vectors/matrices, or NumPy array.

    Returns
    -------
    number, list or numpy.ndarray
        Sum along the outermost dimension, preserving nested list/array
        representation (except 1D inputs, which return a scalar).

    Raises
    ------
    ValueError
        If nested lists have inconsistent shapes, or the array is scalar.
    TypeError
        If elements are not compatible with addition.

    Examples
    --------
    >>> sum_lists([[1, 2], [3, 4]])
    [4, 6]
    >>> sum_lists([[[1, 2]], [[3, 4]]])
    [[4, 6]]
    """
    if isinstance(lists, np.ndarray):
        if lists.ndim == 0:
            raise ValueError("'lists' must have at least one dimension.")
        return np.sum(lists, axis=0)
    if not isinstance(lists, list):
        raise TypeError("'lists' must be a list or NumPy array.")
    if not lists:
        return []

    def sum_nested(items: list[Any]) -> Any:
        is_list = [isinstance(item, list) for item in items]
        if any(is_list):
            if not all(is_list):
                raise ValueError("Nested lists must have consistent shapes.")
            length = len(items[0])
            if any(len(item) != length for item in items):
                raise ValueError("Nested lists must have consistent shapes.")
            return [sum_nested([item[index] for item in items])
                    for index in range(length)]
        return sum(items)

    return sum_nested(lists)


def list_gcd(values: list[NestedNumbers] | NDArray[Any]) -> int:
    """Calculate the GCD of all numbers in a list, vector, or matrix.

    Nested lists and NumPy arrays are flattened. Floating-point values are
    rounded using Python's ``round()`` (ties to even) before calculation.

    Parameters
    ----------
    values : list or numpy.ndarray
        Integer or real-valued numbers, possibly in nested lists/matrices.

    Returns
    -------
    int
        Greatest common divisor of all rounded values.

    Raises
    ------
    ValueError
        If no values are supplied.
    TypeError
        If a value cannot be rounded to an integer.

    Examples
    --------
    >>> list_gcd([[6, 12], [18, 24]])
    6
    """
    if isinstance(values, np.ndarray):
        flattened = values.flat
    elif isinstance(values, list):
        def flatten(items: list[Any]):
            for item in items:
                if isinstance(item, list):
                    yield from flatten(item)
                else:
                    yield item
        flattened = flatten(values)
    else:
        raise TypeError("'values' must be a list or NumPy array.")

    integers = [int(round(value)) for value in flattened]
    if not integers:
        raise ValueError("'values' must not be empty.")
    return gcd(*integers)


def _as_array(values: list[Any] | NDArray[Any]) -> NDArray[Any]:
    """Convert regular Python lists to non-scalar NumPy arrays."""
    if not isinstance(values, (list, np.ndarray)):
        raise TypeError("'values' must be a list or NumPy array.")
    try:
        array = np.asarray(values)
    except ValueError as exc:
        raise ValueError("Nested lists must have rectangular shapes.") from exc
    if array.ndim == 0:
        raise ValueError("Input must have at least one dimension.")
    return array


def _positions(mask: NDArray[np.bool]) -> NDArray[np.intp]:
    """Return 1D indices for vectors, coordinate rows otherwise."""
    if mask.ndim == 1:
        return np.flatnonzero(mask)
    return np.argwhere(mask)


def _first_index(flat_index: int, shape: tuple[int, ...]) -> Position:
    """Return a scalar index for vectors, coordinate tuple otherwise."""
    if len(shape) == 1:
        return flat_index
    return tuple(int(pos) for pos in np.unravel_index(flat_index, shape))


def select_value(
    vector: list[Any] | NDArray[Any],
    value: Any,
    absolute_tolerance: float = 1e-6,
    first_only: bool = False,
    relative_tolerance: float = 0.0,
) -> tuple[NDArray[Any], NDArray[np.intp]] | tuple[Any, Position]:
    """Select values approximately equal to a target, with their positions.

    Numeric arrays are compared with ``np.isclose``. Text, dates, and other
    non-numeric data use exact equality. Supports lists, rectangular nested
    lists, vectors, and matrices (and higher-dimensional arrays).

    Parameters
    ----------
    vector : list or numpy.ndarray
        Values to search, in any non-scalar dimensionality.
    value : object
        Target value to match.
    absolute_tolerance : float, optional
        Absolute comparison tolerance (default: 1e-6).
    first_only : bool, optional
        If True, return only the first matching element and its position.
    relative_tolerance : float, optional
        Relative comparison tolerance (default: 0.0).

    Returns
    -------
    tuple
        Selected values (1D NumPy array) and positions. Vector positions
        are integer indices; matrix positions are ``(row, column)`` pairs
        in an ``(n, 2)`` array. With ``first_only=True``, returns a scalar
        value and an integer index or a tuple of coordinates.

    Raises
    ------
    ValueError
        If tolerances are negative, nested lists are ragged, or
        ``first_only=True`` but no match exists.

    Examples
    --------
    >>> select_value([[1, 2], [2, 3]], 2)[1].tolist()
    [[0, 1], [1, 0]]
    """
    array = _as_array(vector)
    if absolute_tolerance < 0 or relative_tolerance < 0:
        raise ValueError("Comparison tolerances must be non-negative.")

    if np.issubdtype(array.dtype, np.number):
        mask = np.isclose(
            array, value, atol=absolute_tolerance, rtol=relative_tolerance
        )
    else:
        mask = array == value

    indices = _positions(mask)
    selected = array[mask]
    if first_only:
        if selected.size == 0:
            raise ValueError(f"No value matching {value!r} was found.")
        flat = int(np.flatnonzero(mask)[0])
        return selected[0], _first_index(flat, array.shape)
    return selected, indices


def select_closest_value(
    vector: list[Any] | NDArray[Any],
    value: Any,
) -> tuple[Any, Position]:
    """Find the element closest to a target and its position.

    Supports numerical lists, vectors, matrices, and datetimes supporting
    subtraction/absolute differences. For ties, the first (row-major)
    occurrence is selected. NaN distances are ignored.

    Parameters
    ----------
    vector : list or numpy.ndarray
        Non-empty, non-scalar values to search.
    value : object
        Target compatible with subtraction from values.

    Returns
    -------
    tuple
        Closest scalar value and its integer index (vector) or tuple
        of coordinates (matrix/multidimensional data).

    Raises
    ------
    ValueError
        If the input is empty or all distances are NaN.
    TypeError
        If a distance cannot be computed.

    Examples
    --------
    >>> select_closest_value([[1, 4], [8, 10]], 7)
    (np.int64(8), (1, 0))
    """
    array = _as_array(vector)
    if array.size == 0:
        raise ValueError("Input must not be empty.")
    distances = np.abs(array - value)
    if np.issubdtype(distances.dtype, np.floating):
        nan_mask = np.isnan(distances)
        if nan_mask.all():
            raise ValueError("All distances are NaN.")
        distances = np.where(nan_mask, np.inf, distances)
    flat = int(np.argmin(distances))
    return array.flat[flat], _first_index(flat, array.shape)


def select_values_in_range(
    vector: list[Any] | NDArray[Any],
    min_value: Any,
    max_value: Any,
    step: int = 1,
    number: int | None = None,
) -> tuple[NDArray[Any], NDArray[np.intp]]:
    """Select elements within an inclusive interval with their positions.

    Supports numerical and other comparable data (including dates), in
    vectors and matrices. Positions use row-major order. ``number`` takes
    precedence over ``step`` and selects approximately even positions.

    Parameters
    ----------
    vector : list or numpy.ndarray
        Input values; nested Python lists must be rectangular.
    min_value : object
        Inclusive lower bound.
    max_value : object
        Inclusive upper bound.
    step : int, optional
        Select every ``step``-th match (default: 1).
    number : int, optional
        Maximum number of approximately evenly spaced matches to return.
        Overrides ``step`` if given.

    Returns
    -------
    tuple of numpy.ndarray
        Selected values (1D array) and indices. Vector indices are 1D;
        multidimensional indices have one coordinate column per dimension.
        Empty selections produce empty arrays with the correct shapes.

    Raises
    ------
    ValueError
        If ``min_value > max_value``, ``step`` or ``number`` is invalid,
        or nested input lists have inconsistent dimensions.

    Examples
    --------
    >>> select_values_in_range([[1, 2], [3, 4]], 2, 3)[1].tolist()
    [[0, 1], [1, 0]]
    """
    array = _as_array(vector)
    if min_value > max_value:
        raise ValueError("'min_value' must not exceed 'max_value'.")
    if number is not None:
        if isinstance(number, (bool, np.bool_)) or not isinstance(number, (int, np.integer)) or number < 1:
            raise ValueError("'number' must be a positive integer.")
    elif isinstance(step, (bool, np.bool_)) or not isinstance(step, (int, np.integer)) or step < 1:
        raise ValueError("'step' must be a positive integer.")

    flat_matches = np.flatnonzero((array >= min_value) & (array <= max_value))
    if number is not None and number < flat_matches.size:
        chosen = np.linspace(0, flat_matches.size - 1, number, dtype=int)
        flat_matches = flat_matches[chosen]
    elif number is None:
        flat_matches = flat_matches[::step]

    selected = array.flat[flat_matches]
    if array.ndim == 1:
        return selected, flat_matches
    coordinates = np.column_stack(np.unravel_index(flat_matches, array.shape))
    return selected, coordinates.astype(np.intp, copy=False)


def select_row_or_column[T](
    matrix: list[list[T]] | NDArray[Any],
    index: int,
    axis: int = 0,
) -> list[T] | NDArray[Any]:
    """Select a row or column from a Python-list or NumPy matrix.

    Negative indices follow regular Python indexing rules.
    List matrices must be rectangular when a column is selected.

    Parameters
    ----------
    matrix : list of lists or numpy.ndarray
        Two-dimensional matrix.
    index : int
        Row (``axis=0``) or column (``axis=1``) index.
    axis : {0, 1}, optional
        0 selects a row (default); 1 selects a column.

    Returns
    -------
    list or numpy.ndarray
        Selected row or column, preserving list/NumPy representation.

    Raises
    ------
    TypeError
        If the input is not a matrix or indices are not integers.
    ValueError
        If the matrix shape is invalid or ``axis`` is not 0 or 1.
    IndexError
        If ``index`` falls outside the corresponding dimension.

    Examples
    --------
    >>> select_row_or_column([[1, 2], [3, 4]], 1, axis=1)
    [2, 4]
    """
    if isinstance(axis, bool) or axis not in (0, 1):
        raise ValueError("'axis' must be 0 (row) or 1 (column).")
    if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
        raise TypeError("'index' must be an integer.")

    if isinstance(matrix, np.ndarray):
        if matrix.ndim != 2:
            raise ValueError("'matrix' must be two-dimensional.")
        return matrix[index, :] if axis == 0 else matrix[:, index]

    if not isinstance(matrix, list) or any(not isinstance(row, list) for row in matrix):
        raise TypeError("'matrix' must be a list of lists or a 2D NumPy array.")
    if axis == 0:
        return matrix[index]
    if not matrix:
        raise IndexError("Cannot select a column from an empty matrix.")
    column_count = len(matrix[0])
    if any(len(row) != column_count for row in matrix):
        raise ValueError("All rows must have equal lengths to select a column.")
    if not (-column_count <= index < column_count):
        raise IndexError(f"Column index {index} is out of range.")
    return [row[index] for row in matrix]
