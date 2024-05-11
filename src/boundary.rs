use crate::pixelgrid::PixelGrid;
use crate::fluid_state::FluidState;
use crate::particle::Particle;

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


pub fn get_ghost_particles_naive(fs: &FluidState, pg: &PixelGrid) -> Vec<Particle> {
    let mut res = vec![];
    for i in 0..pg.m {
        for j in 0..pg.n {
            let ak = i * pg.n + j;
            if fs.boundary[ak] == 0.0 {
                res.push(Particle{
                    position: (j as f32 + pg.dx / 2.0, i as f32 + pg.dy / 2.0),
                    mass: 1.0,
                    density: 1.0,
                    pressure: 0.0,
                    .. Default::default()
                })
            }
        }
    }
    res
}

pub fn get_ghost_box(fs: &mut FluidState, pg: &PixelGrid, mass: f32, i0: i32, ie: i32, j0: i32, je: i32, n: i32) -> Vec<Particle> {
    let mut res = vec![];
    for i in i0..ie {
        let (x, y) = pg.worldxy2xy(j0 as f32, i as f32);
        let ak = pg.xy2ak(x, y);
        fs.boundary[ak] = 0.0;
        let (x, y) = pg.worldxy2xy(je as f32, i as f32);
        let ak = pg.xy2ak(x, y);
        fs.boundary[ak] = 0.0;
        // res.push(Particle{
        //     position: (j0 as f32 + pg.dx / 2.0, i as f32 + pg.dy / 2.0),
        //     mass: 1.0,
        //     density: 1.0,
        //     pressure: 1.0,
        //     .. Default::default()
        // });
        // res.push(Particle{
        //     position: (je as f32 + pg.dy / 2.0, i as f32 + pg.dy / 2.0),
        //     mass: 1.0,
        //     density: 1.0,
        //     pressure: 1.0,
        //     .. Default::default()
        // });
    }
    for j in j0..je + 1 {
        let (x, y) = pg.worldxy2xy(j as f32, i0 as f32);
        let ak = pg.xy2ak(x, y);
        fs.boundary[ak] = 0.0;
        let (x, y) = pg.worldxy2xy(j as f32, ie as f32);
        let ak = pg.xy2ak(x, y);
        fs.boundary[ak] = 0.0;
        // res.push(Particle{
        //     position: (j as f32 + pg.dx / 2.0, i0 as f32 + pg.dy / 2.0),
        //     mass: 1.0,
        //     density: 1.0,
        //     pressure: 1.0,
        //     .. Default::default()
        // });
        // res.push(Particle{
        //     position: (j as f32 + pg.dx / 2.0, ie as f32 + pg.dy / 2.0),
        //     mass: 1.0,
        //     density: 1.0,
        //     pressure: 1.0,
        //     .. Default::default()
        // });
    }
    res
}