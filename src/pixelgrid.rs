pub struct PixelGrid {
    pub m: usize,
    pub n: usize,
    pub mn: usize,
    pub dx: f32,
    pub dy: f32,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl PixelGrid {
    pub fn new(_m: usize, _n: usize) -> Self {
        PixelGrid {
            m: _m,
            n: _n,
            mn: _m * _n,
            dx: 1.0,
            dy: 1.0,
            x: 0.0,
            y: 0.0,
            w: 1.0 * _n as f32,
            h: 1.0 * _m as f32
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
    pub fn sample(&self, data: &Vec<f32>, x: f32, y: f32) -> f32 {
        data[self.worldxy2ak(x, y) as usize]
    }
    pub fn worldxy2ak(&self, x: f32, y: f32) -> usize {
        let x_grid = (x - self.x) as usize;
        let y_grid = (y - self.y) as usize;
//        let x_rel = self.n * x_grid / self.w;
//        let y_rel = self.m * y_grid / self.h;
//        let ak = y_rel * self.n as f32 + x_rel; 
        let ak = y_grid * self.n + x_grid;
        //println!("{} {} => {} {} => {}", x_grid, y_grid, x_rel, y_rel, ak);
        ak 
    }
}
