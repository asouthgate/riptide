use crate::boundary;

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
}