pub struct PixelGrid {
    pub m: usize,
    pub n: usize,
    pub mn: usize,
    pub dx: f32,
    pub dy: f32
}

impl PixelGrid {
    pub fn new(_m: usize, _n: usize) -> Self {
        PixelGrid {
            m: _m,
            n: _n,
            mn: _m * _n,
            dx: 1.0,
            dy: 1.0
        }
    }
    pub fn print_data(&self, data: &Vec<f32>) {
        let prec = 3;
        let underscores = "-".repeat(( 2 + prec + 1) * self.n);
        println!("-{}", underscores);
        for i in 0..self.m {
            print!("|");
            for j in 0..self.n {
                let f = format!("{:.precision$}", data[i * self.n + j], precision = prec);
                print!("{}|", f);
            }
            println!();
        }
        println!("-{}", underscores);
    }
}