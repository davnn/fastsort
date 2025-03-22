use std::mem::MaybeUninit;

use numpy::{
    ndarray::{Array, Array2, ArrayViewD, Axis},
    Element, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods, ToPyArray,
};
use pyo3::prelude::*;
use rayon::prelude::*;

/// An enum representing the different supported array types for sorting in Python.
///
/// This enum is used to differentiate between various numpy array types (e.g., `f64`, `i32`, etc.)
/// and allows for specialized sorting functions to be applied based on the type of array passed
/// to the Python function.
#[derive(FromPyObject)]
enum SupportedArray<'py> {
    F64(PyReadonlyArrayDyn<'py, f64>),
    F32(PyReadonlyArrayDyn<'py, f32>),
    I64(PyReadonlyArrayDyn<'py, i64>),
    I32(PyReadonlyArrayDyn<'py, i32>),
    I16(PyReadonlyArrayDyn<'py, i16>),
    U64(PyReadonlyArrayDyn<'py, u64>),
    U32(PyReadonlyArrayDyn<'py, u32>),
    U16(PyReadonlyArrayDyn<'py, u16>),
}

/// A macro to generate sorting functions for each supported array type.
///
/// Since Rust does not allow generic return types for Python bindings, we use this macro to create
/// sorting functions for specific types (e.g., `f64`, `i32`). The macro calls the generic
/// `sort_generic` function to handle the sorting logic.
macro_rules! generate_sort_function {
    ($type:ty, $name:ident) => {
        #[pyfunction]
        fn $name<'py>(
            py: Python,
            arr: PyReadonlyArrayDyn<$type>,
            axis: Option<isize>,
        ) -> PyResult<Py<PyArrayDyn<$type>>> {
            sort_generic(py, arr, axis)
        }
    };
}

// Generate sorting functions for different types
generate_sort_function!(f64, sortf64);
generate_sort_function!(f32, sortf32);
generate_sort_function!(i64, sorti64);
generate_sort_function!(i32, sorti32);
generate_sort_function!(i16, sorti16);
generate_sort_function!(u64, sortu64);
generate_sort_function!(u32, sortu32);
generate_sort_function!(u16, sortu16);

/// A generic function to sort a numpy array in Python.
///
/// This function handles both 1D arrays and higher-dimensional arrays, and sorts the data
/// either by vector or by array based on the given axis.
fn sort_generic<T: Element + Clone + PartialOrd>(
    py: Python,
    arr: PyReadonlyArrayDyn<'_, T>,
    axis: Option<isize>,
) -> PyResult<Py<PyArrayDyn<T>>> {
    if arr.ndim() == 1 || axis.is_none() {
        sort_vector(py, arr.to_vec().unwrap())
    } else {
        sort_array(py, arr.as_array(), axis.unwrap())
    }
}

/// A helper function to sort a copy of a 1D vector.
///
/// This function sorts the provided vector using parallel sorting and returns the sorted result
/// as a numpy array.
fn sort_vector<T: Element + Clone + PartialOrd>(
    py: Python,
    mut arr: Vec<T>,
) -> PyResult<Py<PyArrayDyn<T>>> {
    arr.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(arr.to_pyarray(py).to_dyn().clone().unbind())
}

/// A helper function to sort a higher-dimensional array along a specified axis.
///
/// This function handles multi-dimensional arrays, using parallelism to efficiently sort each lane
/// along the specified axis and then permuting the result back to the original shape.
fn sort_array<T: Element + Clone + PartialOrd>(
    py: Python,
    array: ArrayViewD<'_, T>,
    axis: isize,
) -> PyResult<Py<PyArrayDyn<T>>> {
    let dim = array.ndim();
    let axis = determine_axis(axis, dim);

    // Move sort axis to last axis
    let permute_sort = permute_sort_axis(dim, axis, false);
    let permute_orig = permute_sort_axis(dim, axis, true);
    let permuted = array.permuted_axes(permute_sort);
    let mut result = permuted.as_standard_layout();

    // The lanes now always refer to the last axis
    let res_lane = result.lanes_mut(Axis(dim - 1));

    // Parallel iteration over lanes
    res_lane.into_iter().par_bridge().for_each(|mut res_slice| {
        let res_mut = res_slice.as_slice_mut().unwrap();
        res_mut.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    });

    // Permute back to original shape
    Ok(result
        .permuted_axes(permute_orig)
        .to_pyarray(py)
        .to_dyn()
        .clone()
        .unbind())
}

/// A helper function to permute the axis of the array for sorting.
///
/// This function ensures that the sorting axis is moved to the last position and returns the
/// modified axis order. It can also reverse the permutation.
fn permute_sort_axis(ndim: usize, axis: usize, reverse: bool) -> Vec<usize> {
    let mut order: Vec<usize> = (0..ndim).collect();
    if reverse {
        order.swap(ndim - 1, axis); // Swap back to original axis position
    } else {
        order.swap(axis, ndim - 1); // Move sorting axis to last position
    }
    order
}

/// A Python function that performs argsort on various array types.
///
/// This function receives a `SupportedArray` and uses the appropriate sorting function
/// to return the sorted indices as a numpy array.
#[pyfunction]
fn argsort<'py>(
    py: Python<'py>,
    arr: SupportedArray<'py>,
    axis: Option<isize>,
) -> PyResult<Py<PyArrayDyn<i64>>> {
    match arr {
        SupportedArray::F64(array) => argsort_generic(py, array, axis),
        SupportedArray::F32(array) => argsort_generic(py, array, axis),
        SupportedArray::I64(array) => argsort_generic(py, array, axis),
        SupportedArray::I32(array) => argsort_generic(py, array, axis),
        SupportedArray::I16(array) => argsort_generic(py, array, axis),
        SupportedArray::U64(array) => argsort_generic(py, array, axis),
        SupportedArray::U32(array) => argsort_generic(py, array, axis),
        SupportedArray::U16(array) => argsort_generic(py, array, axis),
    }
}

/// A generic function to perform argsort on a numpy array.
///
/// This function sorts the array or vector and returns the indices of the sorted elements.
fn argsort_generic<T: Element + PartialOrd>(
    py: Python,
    arr: PyReadonlyArrayDyn<'_, T>,
    axis: Option<isize>,
) -> PyResult<Py<PyArrayDyn<i64>>> {
    if arr.ndim() == 1 || axis.is_none() {
        argsort_vector(py, arr.to_vec().unwrap())
    } else {
        argsort_array(py, arr.as_array(), axis.unwrap())
    }
}

/// A helper function to perform argsort on a 1D vector.
///
/// This function returns the indices of the sorted elements of the vector.
fn argsort_vector<T: Element + PartialOrd>(
    py: Python,
    arr: Vec<T>,
) -> PyResult<Py<PyArrayDyn<i64>>> {
    let mut indices = (0..arr.len() as i64).collect::<Vec<_>>();
    indices.par_sort_unstable_by(|&a, &b| arr[a as usize].partial_cmp(&arr[b as usize]).unwrap());
    Ok(indices.to_pyarray(py).to_dyn().clone().unbind())
}

/// A helper function to perform argsort on a multi-dimensional array along a specified axis.
///
/// This function computes the indices that would sort the array along the given axis and returns
/// them as a numpy array.
fn argsort_array<T: Element + PartialOrd + Send + Sync>(
    py: Python,
    array: ArrayViewD<'_, T>,
    axis: isize,
) -> PyResult<Py<PyArrayDyn<i64>>> {
    let axis = determine_axis(axis, array.ndim());

    // Create indices for sorting and result array (avoids frequent reallocations)
    let len_lane = array.len_of(Axis(axis));
    let num_lanes = (array.len() / len_lane).into();
    let mut indices = Array2::from_shape_fn((num_lanes, len_lane), |(_, col)| col as i64);
    let mut result = Array::<i64, _>::uninit(array.raw_dim());

    // Create iterators over data
    let arr_lane_iter = array.lanes(Axis(axis));
    let res_lane_iter = result.lanes_mut(Axis(axis)).into_iter();
    let idx_row_iter = indices.rows_mut();

    // Parallel iteration over the input array, the result array and the indices
    res_lane_iter
        .zip(arr_lane_iter)
        .zip(idx_row_iter)
        .par_bridge()
        .for_each(|((mut res_slice, arr_slice), mut idx_slice)| {
            // Sort the pre-allocated index slice
            let idx_mut = idx_slice.as_slice_mut().unwrap();
            idx_mut.sort_unstable_by(|&a, &b| {
                arr_slice[a as usize]
                    .partial_cmp(&arr_slice[b as usize])
                    .unwrap()
            });

            // Fill the uninitialized result slice with values
            res_slice
                .iter_mut()
                .zip(idx_mut)
                .for_each(|(mut uninit, elem)| {
                    MaybeUninit::write(&mut uninit, *elem);
                });
        });

    Ok(unsafe { result.assume_init().to_pyarray(py).into() })
}

/// A helper function to determine the axis for sorting, accounting for negative indices.
///
/// This function ensures the axis is valid, even for negative indices, by adjusting them to
/// the correct dimension index.
fn determine_axis(axis: isize, ndim: usize) -> usize {
    if axis < 0 {
        (ndim as isize + axis) as usize
    } else {
        axis as usize
    }
}

/// A Python module implemented in Rust.
///
/// This module provides the sorting functions (`sort` and `argsort`) for various numpy array types
/// and serves as the entry point for Python to interact with the implemented functionality.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sortf64, m)?)?;
    m.add_function(wrap_pyfunction!(sortf32, m)?)?;
    m.add_function(wrap_pyfunction!(sorti64, m)?)?;
    m.add_function(wrap_pyfunction!(sorti32, m)?)?;
    m.add_function(wrap_pyfunction!(sorti16, m)?)?;
    m.add_function(wrap_pyfunction!(sortu64, m)?)?;
    m.add_function(wrap_pyfunction!(sortu32, m)?)?;
    m.add_function(wrap_pyfunction!(sortu16, m)?)?;
    m.add_function(wrap_pyfunction!(argsort, m)?)?;
    Ok(())
}
