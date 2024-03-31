use crate::boundary;
use crate::pixelgrid::PixelGrid;
use crate::momentum::{cal_new_velocity_boundary_aware_no_diffusion, cal_new_q_boundary_aware_no_diffusion};


pub struct FluidState {
    pub u: Vec<f32>,
    pub v: Vec<f32>,
    pub newu: Vec<f32>,
    pub newv: Vec<f32>,
    pub boundary: Vec<f32>, // boundary are central pixels; 1 boundary can affect 4 velocities
    pub boundary_x: Vec<f32>, // whether some x value is boundary adjacent
    pub boundary_y: Vec<f32>,
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
    let mut ak: usize;
    let mut is_not_boundary_x: f32;
    let mut is_not_boundary_y: f32;
    for i in 1..pg.m {
        for j in 1..pg.n {
            ak = i * pg.n + j;
            is_not_boundary_x = fs.boundary[ak].min(fs.boundary[ak-1]);
            is_not_boundary_y = fs.boundary[ak].min(fs.boundary[ak-pg.n]);
            fs.dpdx[ak] = is_not_boundary_x * (fs.pressure[ak] - fs.pressure[ak - 1]) / pg.dx; 
            fs.dpdy[ak] = is_not_boundary_y * (fs.pressure[ak] - fs.pressure[ak - pg.n]) / pg.dy;
        }
    }
}


impl FluidState {
    pub fn new(pg: &PixelGrid) -> Self {
        let mn = pg.mn;
        let zeros = vec![0.0; mn];
        let mut tmp = FluidState {
            u: zeros.clone(),
            v: zeros.clone(),
            newu: zeros.clone(),
            newv: zeros.clone(),
            boundary: zeros.clone(),
            boundary_x: zeros.clone(),
            boundary_y: zeros.clone(),
            divergence: zeros.clone(),
            pressure: zeros.clone(),
            dpdx: zeros.clone(),
            dpdy: zeros.clone(),
            laplacian: zeros.clone()
        };
        boundary::initialize_square_boundary(&mut tmp, pg);
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
        self.swap_vectors();
    }

    pub fn quantity_step(&mut self, fs: &FluidState, pg: &PixelGrid, dt: f32) {
        for ak in 0..pg.mn { 
            cal_new_q_boundary_aware_no_diffusion(
                self, &fs, &pg, ak, dt
            );
        }
        self.swap_vectors();
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

    pub fn set_boundary(&mut self, pg: &PixelGrid, ak: usize, val: f32) {
        self.boundary[ak] = val;
        let j = ak % pg.n;
        if j > 0 {
            self.boundary_x[ak] = self.boundary[ak].min(self.boundary[ak-1]);
        }
        if j < pg.n - 1 {
            self.boundary_x[ak + 1] = self.boundary[ak + 1].min(self.boundary[ak]);
        }
        if ak > pg.n {
            self.boundary_y[ak] = self.boundary[ak].min(self.boundary[ak-pg.n]);
        }
        if ak < pg.mn - pg.n {
            self.boundary_y[ak + pg.n] = self.boundary[ak + pg.n].min(self.boundary[ak]);
        }
    }

}
