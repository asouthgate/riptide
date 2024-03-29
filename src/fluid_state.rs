use crate::boundary;
use crate::pixelgrid::PixelGrid;
use crate::momentum::cal_new_velocity_boundary_aware_no_diffusion;

pub struct FluidState {
    pub u: Vec<f32>,
    pub v: Vec<f32>,
    pub newu: Vec<f32>,
    pub newv: Vec<f32>,
    pub boundary: Vec<f32>,
    pub divergence: Vec<f32>,
    pub pressure: Vec<f32>, // current estimate of pressure field
    pub dpdx: Vec<f32>, // est. pressure correction in x
    pub dpdy: Vec<f32>, // est. pressure correction in y
    pub laplacian: Vec<f32> // divergence of corrections
}


pub fn cal_div(
    u: &Vec<f32>, 
    v: &Vec<f32>,
    result_div: &mut Vec<f32>,
    pg: &PixelGrid
) {
    for i in 0..pg.m-1 {
        for j in 0..pg.n-1 {
            let ak: usize = i * pg.n + j;
            let dudx = (u[ak + 1] - u[ak]) / pg.dx;
            let dvdy = (v[ak + pg.n] - v[ak]) / pg.dy;
            result_div[ak] =  dudx + dvdy;
        }
    }
}


pub fn cal_pressure_corrections(
    fs: &mut FluidState, 
    pg: &PixelGrid
) {
    for i in 1..pg.m {
        for j in 1..pg.n {
            let ak: usize = i * pg.n + j;

            let mut is_boundary_x = fs.boundary[ak] == 0.0;
            let mut is_boundary_y = fs.boundary[ak] == 0.0;

            if ak > 0 {
                is_boundary_x = is_boundary_x || (fs.boundary[ak-1] == 0.0);
            }
            if ak > pg.n {
                is_boundary_y = is_boundary_y || (fs.boundary[ak-pg.n] == 0.0);
            }
            if !is_boundary_x { 
                fs.dpdx[ak] = (fs.pressure[ak] - fs.pressure[ak - 1]) / pg.dx;
            }
            if !is_boundary_y { 
                fs.dpdy[ak] = (fs.pressure[ak] - fs.pressure[ak - pg.n]) / pg.dx;
            }
        }
    }
}


impl FluidState {
    pub fn new(m: usize, n: usize) -> Self {
        let mn = m * n;
        let zeros = vec![0.0; mn];
        let mut tmp = FluidState {
            u: zeros.clone(),
            v: zeros.clone(),
            newu: zeros.clone(),
            newv: zeros.clone(),
            boundary: zeros.clone(),
            divergence: zeros.clone(),
            pressure: zeros.clone(),
            dpdx: zeros.clone(),
            dpdy: zeros.clone(),
            laplacian: zeros.clone()
        };
        boundary::initialize_square_boundary(&mut tmp.boundary, m, n);
        tmp
    }

    pub fn swap_vectors(&mut self) {
        std::mem::swap(&mut self.u, &mut self.newu);
        std::mem::swap(&mut self.v, &mut self.newv);
    }

    pub fn momentum_step(&mut self, pg: &PixelGrid, dt: f32) {
        let mut ak: usize;
        for i in 1..pg.m {
            for j in 1..pg.n {
                ak = i * pg.n + j;
                cal_new_velocity_boundary_aware_no_diffusion(self, pg, ak, dt);
            }
        }
    }

    pub fn apply_corrections(&mut self) {
        for ak in 0..self.dpdx.len() {
            self.u[ak] -= self.dpdx[ak];
            self.v[ak] -= self.dpdy[ak];
        }
    }

    pub fn cal_divergence(&mut self, pg: &PixelGrid) {
        cal_div(&self.u, &self.v, &mut self.divergence, pg)
    }

}