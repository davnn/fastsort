use std::mem::MaybeUninit;

use numpy::{
    ndarray::{Array, Array2, ArrayViewD, Axis},
    Element, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods, ToPyArray,
};
use pyo3::prelude::*;
use rayon::prelude::*;

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

// it appears to be impossible to use generic return types for python, instead
// we generate sort functions for python using a macro that calls a generic
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

generate_sort_function!(f64, sortf64);
generate_sort_function!(f32, sortf32);
generate_sort_function!(i64, sorti64);
generate_sort_function!(i32, sorti32);
generate_sort_function!(i16, sorti16);
generate_sort_function!(u64, sortu64);
generate_sort_function!(u32, sortu32);
generate_sort_function!(u16, sortu16);

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

fn sort_vector<T: Element + Clone + PartialOrd>(
    py: Python,
    mut arr: Vec<T>,
) -> PyResult<Py<PyArrayDyn<T>>> {
    arr.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(arr.to_pyarray(py).to_dyn().clone().unbind())
}

fn sort_array<T: Element + Clone + PartialOrd>(
    py: Python,
    array: ArrayViewD<'_, T>,
    axis: isize,
) -> PyResult<Py<PyArrayDyn<T>>> {
    let dim = array.ndim();
    let axis = determine_axis(axis, dim);

    // move sort axis to last axis
    let permute_sort = permute_sort_axis(dim, axis, false);
    let permute_orig = permute_sort_axis(dim, axis, true);
    let permuted = array.permuted_axes(permute_sort);
    let mut result = permuted.as_standard_layout();

    // the lanes now always refer to the last axis
    let res_lane = result.lanes_mut(Axis(dim - 1));

    // parallel iteration over lanes
    res_lane.into_iter().par_bridge().for_each(|mut res_slice| {
        let res_mut = res_slice.as_slice_mut().unwrap();
        res_mut.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    });

    // permute back to original shape
    Ok(result
        .permuted_axes(permute_orig)
        .to_pyarray(py)
        .to_dyn()
        .clone()
        .unbind())
}

fn permute_sort_axis(ndim: usize, axis: usize, reverse: bool) -> Vec<usize> {
    let mut order: Vec<usize> = (0..ndim).collect();
    if reverse {
        order.swap(ndim - 1, axis); // Swap back to original axis position
    } else {
        order.swap(axis, ndim - 1); // Move sorting axis to last position
    }
    order
}

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

fn argsort_vector<T: Element + PartialOrd>(
    py: Python,
    arr: Vec<T>,
) -> PyResult<Py<PyArrayDyn<i64>>> {
    let mut indices = (0..arr.len() as i64).collect::<Vec<_>>();
    indices.par_sort_unstable_by(|&a, &b| arr[a as usize].partial_cmp(&arr[b as usize]).unwrap());
    Ok(indices.to_pyarray(py).to_dyn().clone().unbind())
}

fn argsort_array<T: Element + PartialOrd + Send + Sync>(
    py: Python,
    array: ArrayViewD<'_, T>,
    axis: isize,
) -> PyResult<Py<PyArrayDyn<i64>>> {
    let axis = determine_axis(axis, array.ndim());

    // create indices for sorting and result array (avoids frequent reallocations)
    let len_lane = array.len_of(Axis(axis));
    let num_lanes = (array.len() / len_lane).into();
    let mut indices = Array2::from_shape_fn((num_lanes, len_lane), |(_, col)| col as i64);
    let mut result = Array::<i64, _>::uninit(array.raw_dim());

    // create iterators over data
    let arr_lane_iter = array.lanes(Axis(axis));
    let res_lane_iter = result.lanes_mut(Axis(axis)).into_iter();
    let idx_row_iter = indices.rows_mut();

    // parallel iteration over the input array, the result array and the indices
    res_lane_iter
        .zip(arr_lane_iter)
        .zip(idx_row_iter)
        .par_bridge()
        .for_each(|((mut res_slice, arr_slice), mut idx_slice)| {
            // sort the pre-allocated index slice
            let idx_mut = idx_slice.as_slice_mut().unwrap();
            idx_mut.sort_unstable_by(|&a, &b| {
                arr_slice[a as usize]
                    .partial_cmp(&arr_slice[b as usize])
                    .unwrap()
            });

            // fill the uninitialized result slice with values
            res_slice
                .iter_mut()
                .zip(idx_mut)
                .for_each(|(mut uninit, elem)| {
                    MaybeUninit::write(&mut uninit, *elem);
                });
        });

    Ok(unsafe { result.assume_init().to_pyarray(py).into() })
}

fn determine_axis(axis: isize, ndim: usize) -> usize {
    if axis < 0 {
        (ndim as isize + axis) as usize
    } else {
        axis as usize
    }
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
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
