use crate::boundary;

pub struct FluidState {
    pub u: Vec<f32>,
    pub v: Vec<f32>,
    pub newu: Vec<f32>,
    pub newv: Vec<f32>,
    pub boundary: Vec<f32>
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
            boundary: zeros.clone()
        };
        boundary::initialize_square_boundary(&mut tmp.boundary, m, n);
        tmp
    }
}