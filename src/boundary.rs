/// Set the boundary of an (implicitly) rectangular array to zero.
/// 
/// # Arguments
/// 
/// * `boundary` - The array to fill 
/// * `m` - Number of rows
/// * `n` - Number of columns
pub fn initialize_square_boundary(boundary: &mut Vec<f32>, m: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            boundary[i * n + j] = 1.0;
        }
    }

    for i in 0..m {
        boundary[i * n + 0] = 0.0;
        boundary[i * n + n - 1] = 0.0;
    }

    for j in 0..n {
        boundary[0 * n + j] = 0.0;
        boundary[(m - 1) * n + j] = 0.0;
    }
}