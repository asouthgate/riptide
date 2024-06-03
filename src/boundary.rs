use crate::pixelgrid::PixelGrid;
use crate::fluid_state::FluidState;
use crate::particle::Particle;
use crate::particle_ecs::ParticleData;


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

pub fn get_ghost_box(pg: &PixelGrid, mass: f32, i0: i32, ie: i32, j0: i32, je: i32) -> Vec<Particle> {
    let mut res = vec![];
    for i in i0..ie {
        res.push(Particle{
            position: (j0 as f32, i as f32),
            mass: mass,
            .. Default::default()
        });
        res.push(Particle{
            position: (je as f32, i as f32),
            mass: mass,
            .. Default::default()
        });
        res.push(Particle{
            position: (j0 as f32, i as f32 + pg.dy / 2.0),
            mass: mass,
            .. Default::default()
        });
        res.push(Particle{
            position: (je as f32, i as f32 + pg.dy / 2.0),
            mass: mass,
            .. Default::default()
        });
    }
    for j in j0..je {
        res.push(Particle{
            position: (j as f32, i0 as f32),
            mass: mass,
            .. Default::default()
        });
        res.push(Particle{
            position: (j as f32, ie as f32),
            mass: mass,
            .. Default::default()
        });
        res.push(Particle{
            position: (j as f32 + pg.dx / 2.0, i0 as f32),
            mass: mass,
            .. Default::default()
        });
        res.push(Particle{
            position: (j as f32 + pg.dx / 2.0, ie as f32),
            mass: mass,
            .. Default::default()
        });
    }
    res.push(Particle{
        position: (je as f32, ie as f32),
        mass: mass,
        .. Default::default()
    });
    // res.push(Particle{
    //     position: (je as f32 - 1.x + pg.dx / 2.0, ie as f32),
    //     mass: mass,
    //     .. Default::default()
    // });
    res
}

pub struct SquareBoundary {
    pub i0: f32,
    pub ie: f32,
    pub j0: f32,
    pub je: f32
}

impl SquareBoundary {
    pub fn enforce_boundary_ecs(
        &self,
        pdata: &mut ParticleData<cgmath::Vector2<f32>>,
    ) {
        // first naive case
        for pi in 0..pdata.n_fluid_particles {
            if pdata.x[pi].x < self.j0 {
                pdata.x[pi].x = self.j0;
                pdata.v[pi].x = -pdata.v[pi].x;
            }
            if pdata.x[pi].x > self.je {
                pdata.x[pi].x = self.je; 
                pdata.v[pi].x = -pdata.v[pi].x;
            }
            if pdata.x[pi].y < self.i0 {
                pdata.x[pi].y = self.i0;
                pdata.v[pi].y = -pdata.v[pi].y;
            }
            if pdata.x[pi].y > self.ie {
                pdata.x[pi].y = self.ie;
                pdata.v[pi].y = -pdata.v[pi].y;
            }
        }
    }

}


pub struct HyperbolicSquareBoundary {
    pub i0: f32,
    pub ie: f32,
    pub j0: f32,
    pub je: f32
}


impl HyperbolicSquareBoundary {
    pub fn enforce_boundary_ecs(
        &self,
        pdata: &mut ParticleData<cgmath::Vector2<f32>>,
    ) {
        // first naive case
        for pi in 0..pdata.n_fluid_particles {
            if pdata.x[pi].x <= self.j0 {
                pdata.x[pi].x = self.je - 1.0;
                // pdata.v[pi].x = -pdata.v[pi].x;
            }
            if pdata.x[pi].x >= self.je {
                pdata.x[pi].x = self.j0 + 1.0;
                // pdata.v[pi].x = -pdata.v[pi].x;
            }
            if pdata.x[pi].y <= self.i0 {
                pdata.x[pi].y = self.ie - 1.0;
                // pdata.v[pi].y = -pdata.v[pi].y;
            }
            if pdata.x[pi].y >= self.ie {
                pdata.x[pi].y = self.i0 + 1.0;
                // pdata.v[pi].y = -pdata.v[pi].y;
            }
        }
    }

}
