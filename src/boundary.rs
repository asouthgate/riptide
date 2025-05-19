use crate::pixelgrid::PixelGrid;
use crate::particle_ecs::ParticleData;
use shallow::texture::read_png;
use shallow::texture::image_to_byte_array;
use image::GenericImageView;

pub fn _boundary_pixel(r: u8, g: u8, b: u8, _a: u8) -> bool {
    if r > 20 || g > 20 || b > 20 {
        return true;
    }
    return false;
}

/// Generate a particle boundary from a png image
///
/// # Arguments
///
/// * `pg` - PixelGrid, should have same resolution as image
/// * `fname` - Name of file to read
///
/// # Returns
///
/// A PixelGrid, Vec<(x, y)> tuple, where x, y are 
pub fn read_boundary_from_png(pg: &PixelGrid, fname: String) -> Vec<(f32, f32)> {
    let image = read_png(fname).expect("Failed to read file");
    let dimensions = image.dimensions();
    let m = dimensions.1 as usize;
    let n = dimensions.0 as usize;
    assert_eq!(m, pg.m);
    assert_eq!(n, pg.n);
    let bytes = image_to_byte_array(image);
    let mut ghost_particle_positions = vec![];
    for i in 0..m {
        for j in 0..n {
            let ak: usize = 4 * (i * n  + j);
            if _boundary_pixel(bytes[ak], bytes[ak + 1], bytes[ak + 2], bytes[ak + 3]) {
                let (wx, wy) = pg.ij2wxy(i, j);
                println!("{} {} -> {} {}", i, j, wx, wy);
                let ib = pg.in_bounds_wx(wx, wy);
                if !ib {
                    panic!("Boundary pixel {} {} is not in PixelGrid bounds. Something went very wrong.", wx, wy)
                }
                ghost_particle_positions.push((wx, wy));
            }    
        }
    }
    ghost_particle_positions
}

/// Generate a particle boundary from a png image
///
/// # Arguments
///
/// * `pg` - PixelGrid, should have same resolution as image
/// * `fname` - Name of file to read
///
/// # Returns
///
/// A vector of bits indicating boundary pixels; true for boundary, false otherwise 
pub fn read_boundary_mask_from_png(pg: &PixelGrid, fname: String) -> Vec<bool> {
    let image = read_png(fname).expect("Failed to read file");
    let dimensions = image.dimensions();
    let m = dimensions.1 as usize;
    let n = dimensions.0 as usize;
    assert_eq!(m, pg.m);
    assert_eq!(n, pg.n);
    let bytes = image_to_byte_array(image);
    let mut mask = vec![];
    for i in 0..m {
        for j in 0..n {
            let ak: usize = 4 * (i * n  + j);
            mask.push(_boundary_pixel(bytes[ak], bytes[ak + 1], bytes[ak + 2], bytes[ak + 3]))
        }
    }
    mask
}

pub fn collision(pg: &PixelGrid, mask: &Vec<bool>, wx: (f32, f32)) -> bool {
    return pg.sample_world(&mask, wx.0, wx.1);
}

pub fn enforce_boundary_mask(pg: &PixelGrid, mask: &Vec<bool>, x: &(f32, f32), v: &(f32, f32)) {

}

pub fn get_ghost_box(pg: &PixelGrid, i0: i32, ie: i32, j0: i32, je: i32) -> Vec<(f32, f32)> {
    let mut res = vec![];
    for i in i0..ie {
        res.push((j0 as f32, i as f32));
        res.push((je as f32, i as f32));
        res.push((j0 as f32, i as f32 + pg.dy / 2.0));
        res.push((je as f32, i as f32 + pg.dy / 2.0));
    }
    for j in j0..je {
        res.push((j as f32, i0 as f32));
        res.push((j as f32, ie as f32));
        res.push((j as f32 + pg.dx / 2.0, i0 as f32));
        res.push((j as f32 + pg.dx / 2.0, ie as f32));
    }
    res.push((je as f32, ie as f32));
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
        pdata: &mut ParticleData,
    ) {
        panic!("Deprecated");
        // first naive case
        // for pi in 0..pdata.n_fluid_particles {
        //     if pdata.x[pi].0 <= self.j0 + 1 {
        //         pdata.x[pi].0 = self.j0 + 2;
        //         pdata.v[pi].0 = -pdata.v[pi].0;
        //     }
        //     if pdata.x[pi].0 >= self.je - 1 {
        //         pdata.x[pi].0 <= self.je - 2; 
        //         pdata.v[pi].0 = -pdata.v[pi].0;
        //     }
        //     if pdata.x[pi].1 <= self.i0 + 1 {
        //         pdata.x[pi].1 = self.i0 + 2;
        //         pdata.v[pi].1 = -pdata.v[pi].1;
        //     }
        //     if pdata.x[pi].1 >= self.ie - 1 {
        //         pdata.x[pi].1 = self.ie - 2;
        //         pdata.v[pi].1 = -pdata.v[pi].1;
        //     }
        // }
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
        pdata: &mut ParticleData,
    ) {
        // first naive case
        for pi in 0..pdata.n_fluid_particles {
            if pdata.x[pi].0 < self.j0 {
                pdata.x[pi].0 = self.je;
                // pdata.v[pi].0 = -pdata.v[pi].0;
            }
            if pdata.x[pi].0 > self.je {
                pdata.x[pi].0 = self.j0;
                // pdata.v[pi].0 = -pdata.v[pi].0;
            }
            if pdata.x[pi].1 <= self.i0 {
                pdata.x[pi].1 = self.ie;
                // pdata.v[pi].1 = -pdata.v[pi].1;
            }
            if pdata.x[pi].1 >= self.ie {
                pdata.x[pi].1 = self.i0;
                // pdata.v[pi].1 = -pdata.v[pi].1;
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitmask_collision() {
        let pg = PixelGrid::new(3, 3);
        let mask = vec![
            true, true, true, 
            true, false, true,
            true, true, true
        ];
        let mut x = (1.5, 1.5);
        let mut x_prop = (0.5, 0.5);
        assert!(collision(&pg, &mask, x_prop));
        assert!(!collision(&pg, &mask, (1.5, 1.5)));
        assert!(!collision(&pg, &mask, (1.9, 1.9)));
        assert!(collision(&pg, &mask, (2.1, 1.9)));

    }
}
