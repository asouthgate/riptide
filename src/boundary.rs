use crate::pixelgrid::PixelGrid;
use crate::fluid_state::FluidState;

/// Set the boundary of an (implicitly) rectangular array to zero.
/// 
/// # Arguments
/// 
/// * `m` - Number of rows
/// * `n` - Number of columns
pub fn initialize_square_boundary(fs: &mut FluidState, pg: &PixelGrid) {
    let m = pg.m;
    let n = pg.n;
    for i in 0..m {
        for j in 0..n {
            fs.set_boundary(pg, i * n + j, 1.0);
        }
    }

    for i in 0..m {
        fs.set_boundary(pg, i * n + 0, 0.0);
        fs.set_boundary(pg, i * n + n - 1, 0.0);
    }

    for j in 0..n {
        fs.set_boundary(pg, 0 * n + j, 0.0);
        fs.set_boundary(pg, (m - 1) * n + j, 0.0);
    }
}