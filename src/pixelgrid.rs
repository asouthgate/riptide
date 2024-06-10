pub struct PixelGrid {
    pub m: usize,
    pub n: usize,
    pub l: usize,
    pub mn: usize,
    pub mnl: usize,
    pub dx: f32,
    pub dy: f32,
    pub dz: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
    pub h: f32,
    pub d: f32,
}

impl Default for PixelGrid {
    fn default() -> Self {
        Self {
            m: 100,
            n: 100,
            l: 1,
            mn: 100 * 100,
            mnl: 100 * 100,
            dx: 1.0,
            dy: 1.0,
            dz: 0.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 100.0,
            h: 100.0,
            d: 0.0,
        }
    }
}


impl PixelGrid {

    pub fn new(_m: usize, _n: usize) -> Self {
        PixelGrid {
            m: _m,
            n: _n,
            mn: _m * _n, 
            w: _n as f32,
            h: _m as f32,
            ..Default::default()
        }
    }

    pub fn new_with_transform(
        _m: usize,
        _n: usize,
        _dx: f32,
        _dy: f32,
        _x: f32,
        _y: f32
    ) -> Self {
        PixelGrid {
            m: _m,
            n: _n,
            mn: _m * _n,
            dx: _dx,
            dy: _dy,
            x: _x,
            y: _y,
            w: _dx * _n as f32,
            h: _dy * _m as f32,
            ..Default::default()
        }
    }

    pub fn new3d(_m: usize, _n: usize, _l: usize) -> Self {
        PixelGrid {
            m: _m,
            n: _n,
            l: _l,
            mn: _m * _n,
            mnl: _m * _n * _l,
            w: _n as f32,
            h: _m as f32,
            d: _l as f32,
            dz: 1.0,
            ..Default::default()
        }
    }

    pub fn new3d_with_transform(
        _m: usize,
        _n: usize,
        _l: usize,
        _dx: f32,
        _dy: f32,
        _dz: f32,
        _x: f32,
        _y: f32,
        _z: f32
    ) -> Self {
        PixelGrid {
            m: _m,
            n: _n,
            l: _l,
            mn: _m * _n,
            mnl: _m * _n * _l,
            dx: _dx,
            dy: _dy,
            dz: _dz,
            x: _x,
            y: _y,
            z: _z,
            w: _dx * _n as f32,
            h: _dy * _m as f32,
            d: _dz * _l as f32
        }
    }

    pub fn size(&self) -> usize {
        self. m * self. n * self.l
    }

    pub fn print_data(&self, data: &Vec<f32>) {
        let prec = 3;
        let underscores = "-".repeat(( 2 + prec + 1) * self.n);
        println!("-{}", underscores);
        for k in 0..self.l {
            for i in 0..self.m {
                print!("|");
                for j in 0..self.n {
                    let f = format!("{:.precision$}", data[k * self.mn + i * self.n + j], precision = prec);
                    print!("{}|", f);
                }
                println!();
            }
            println!();
        }
        println!("-{}", underscores);
    }

    pub fn sample_world_3d(&self, data: &Vec<f32>, wx: f32, wy: f32, wz: f32) -> f32 {
        let (x, y, z) = self.worldxyz2xyz(wx, wy, wz);
        self.sample_3d(data, x, y, z)
    }

    pub fn sample_3d(&self, data: &Vec<f32>, x: f32, y: f32, z: f32) -> f32 {
        data[self.xyz2ak(x, y, z)]
    }

    pub fn xyz2ijk(&self, x: f32, y: f32, z: f32) -> Option<(usize, usize, usize)> {
        match (x, y, z) {
            (x, _, _) if (x > self.n as f32 || x < 0.0) => None,
            (_, y, _) if (y > self.m as f32 || y < 0.0) => None,
            (_, _, z) if (z > self.l as f32 || z < 0.0) => None,
            _ => {
                Some((y.trunc() as usize, x.trunc() as usize, z.trunc() as usize))
            }
        }
    }

    pub fn ijk2ak_nocheck(&self, i: usize, j: usize, k: usize) -> usize {
        k * self.mn + i * self.n + j
    }

    pub fn xyz2ak(&self, x: f32, y: f32, z: f32) -> usize {
        let (i, j, k) = self.xyz2ijk(x, y, z).unwrap();
        self.ijk2ak_nocheck(i, j, k)
    }

    pub fn worldxyz2xyz(&self, wx: f32, wy: f32, wz: f32) -> (f32, f32, f32) {
        let x_grid = wx - self.x;
        let y_grid = wy - self.y;
        let z_grid = wz - self.z;
        let x = x_grid / self.dx;
        let y = y_grid / self.dy;
        let z = z_grid / self.dz;
        (x, y, z)
    }

    pub fn xy2ij(&self, x: f32, y:f32) -> Option<(usize, usize)> {
        match (x, y) {
            (x, _) if (x > self.n as f32 || x < 0.0) => None,
            (_, y) if (y > self.m as f32 || y < 0.0) => None,
            _ => {
                Some((y.trunc() as usize, x.trunc() as usize))
            }
        }
    }

    pub fn ij2ak_nocheck(&self, i: usize, j: usize) -> usize {
        i * self.n + j
    }

    pub fn xy2ak(&self, x: f32, y:f32) -> usize {
        let (i, j) = self.xy2ij(x, y).unwrap();
        self.ij2ak_nocheck(i, j)
    }

    pub fn worldxy2xy(&self, wx: f32, wy: f32) -> (f32, f32) {
        let x_grid = wx - self.x;
        let y_grid = wy - self.y;
        let x = x_grid / self.dx;
        let y = y_grid / self.dy;
        (x, y)
    }

    pub fn sample_world(&self, data: &Vec<f32>, wx: f32, wy: f32) -> f32 {
        let (x, y) = self.worldxy2xy(wx, wy);
        self.sample(data, x, y)
    }

    pub fn sample(&self, data: &Vec<f32>, x: f32, y: f32) -> f32 {
        data[self.xy2ak(x, y)]
    }

    pub fn sample_bilinear(&self, data: &Vec<f32>, x: f32, y: f32) -> f32 {
        let xul = x.floor();
        let yul = y.floor() + 1.0;
        let xur = x.floor() + 1.0;
        let yur = y.floor() + 1.0;
        let xbl = x.floor();
        let ybl = y.floor();
        let xbr = x.floor() + 1.0;
        let ybr = y.floor();

        let akul = self.xy2ak(xul, yul);
        let akur = self.xy2ak(xur, yur);
        let akbl = self.xy2ak(xbl, ybl);
        let akbr = self.xy2ak(xbr, ybr);

        let dx = x - xbl;
        let dy = y - ybl;

        let fxy1 = ( data[akbl] * (1.0 - dx) ) + ( data[akbr] * dx );  // recall dx is 1, so we don't divide by width
        let fxy2 = ( data[akul] * (1.0 - dx) ) + ( data[akur] * dx );  // recall dx is 1, so we don't divide by width

        let fxy = fxy1 * (1.0 - dy) + fxy2 * dy;
        return fxy;
    }

    pub fn sample_bilinear_world(&self, data: &Vec<f32>, wx: f32, wy: f32) -> f32 {
        let (x, y) = self.worldxy2xy(wx, wy);
        self.sample_bilinear(data, x, y)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xy2ij_oob() {
        let pg = PixelGrid::new(100, 100);
        let mut result: Option<(usize, usize)> = pg.xy2ij(120.0, 90.0);
        assert_eq!(result, None);
        result = pg.xy2ij(90.0, 101.0);
        assert_eq!(result, None);
        result = pg.xy2ij(-101.0, 90.0);
        assert_eq!(result, None);
        result = pg.xy2ij(-101.0, -900.0);
        assert_eq!(result, None);
    }

    #[test]
    fn test_worldxy2xy_noscaling() {
        let pg = PixelGrid::new(100, 100);
        let (mut x, mut y) = pg.worldxy2xy(90.0, 80.0);
        assert_eq!((x, y), (90.0, 80.0));
        (x, y) = pg.worldxy2xy(-90.0, 80.0);
        assert_eq!((x, y), (-90.0, 80.0));
        (x, y) = pg.worldxy2xy(-90.0, -80.0);
        assert_eq!((x, y), (-90.0, -80.0));
   }

   #[test]
    fn test_worldxy2xy_scaling() {
        let pg = PixelGrid::new_with_transform(100, 100, 0.5, 0.5, 0.0, 0.0);
        let (mut x, mut y) = pg.worldxy2xy(100.0, 100.0);
        assert_eq!((x, y), (200.0, 200.0));
        (x, y) = pg.worldxy2xy(-100.0, -100.0);
        assert_eq!((x, y), (-200.0, -200.0));
    }

    #[test]
    fn test_xyz2ijk_oob() {
        let pg = PixelGrid::new3d(100, 100, 100);
        let mut result: Option<(usize, usize, usize)> = pg.xyz2ijk(120.0, 90.0, 90.0);
        assert_eq!(result, None);
        result = pg.xyz2ijk(90.0, 101.0, 90.0);
        assert_eq!(result, None);
        result = pg.xyz2ijk(-101.0, 90.0, 90.0);
        assert_eq!(result, None);
        result = pg.xyz2ijk(-101.0, -900.0, 90.0);
        assert_eq!(result, None);
        result = pg.xyz2ijk(20.0, 4.0, 110.0);
        assert_eq!(result, None);
    }

    #[test]
    fn test_worldxy2xyz_noscaling() {
        let pg = PixelGrid::new3d(100, 100, 100);
        let (mut x, mut y, mut z) = pg.worldxyz2xyz(90.0, 80.0, 90.0);
        assert_eq!((x, y, z), (90.0, 80.0, 90.0));
        (x, y, z) = pg.worldxyz2xyz(-90.0, 80.0, 50.0);
        assert_eq!((x, y, z), (-90.0, 80.0, 50.0));
        (x, y, z) = pg.worldxyz2xyz(-90.0, -80.0, -20.0);
        assert_eq!((x, y, z), (-90.0, -80.0, -20.0));
   }

   #[test]
    fn test_worldxy2xyz_scaling() {
        let pg = PixelGrid::new3d_with_transform(100, 100, 100, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0);
        let (mut x, mut y, mut z) = pg.worldxyz2xyz(100.0, 100.0, 100.0);
        assert_eq!((x, y, z), (200.0, 200.0, 200.0));
        (x, y, z) = pg.worldxyz2xyz(-100.0, -100.0, -100.0);
        assert_eq!((x, y, z), (-200.0, -200.0, -200.0));
    }


}