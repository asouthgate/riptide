use crate::pixelgrid::PixelGrid;
use crate::vector::Vector;
use cgmath::{Vector2, Vector3};

// For vector2, vector3, anything else
pub trait Indexable {
    fn get_ak(&self, pg: &PixelGrid) -> usize;
    fn get_ijk(&self, pg: &PixelGrid) -> (usize, usize, usize);
    fn get_n_nbr_blocks(&self) -> usize;
}

impl Indexable for Vector2<f32> {
    fn get_ak(&self, pg: &PixelGrid) -> usize {
        let (x, y) = pg.worldxy2xy(self[0], self[1]);
        pg.xy2ak(x, y)    
    }
    fn get_ijk(&self, pg: &PixelGrid) -> (usize, usize, usize) {
        let (x, y) = pg.worldxy2xy(self[0], self[1]);
        let (i, j) = pg.xy2ij(x, y).unwrap();
        (i, j, 0)
    }
    fn get_n_nbr_blocks(&self) -> usize {
        9
    }
}

impl Indexable for Vector3<f32> {
    fn get_ak(&self, pg: &PixelGrid) -> usize {
        let (x, y, z) = pg.worldxyz2xyz(self[0], self[1], self[2]);
        pg.xyz2ak(x, y, z)
    }
    fn get_ijk(&self, pg: &PixelGrid) -> (usize, usize, usize) {
        let (x, y, z) = pg.worldxyz2xyz(self[0], self[1], self[2]);
        pg.xyz2ijk(x, y, z).unwrap()
    }
    fn get_n_nbr_blocks(&self) -> usize {
        27
    }
}

pub struct ParticleIndex<'a> {
    pub start2neighbors: Vec<usize>, // an array with particle indices, implicitly sorted into bins
    ak2start: Vec<usize>, // for each ak, gives index aj of nbr_array, with nbrs
    ak2end: Vec<usize>, // not inclusive, like a 0..end, [a, b)
    pub nbrs: Vec<[&'a [usize]; 27]>, // for precomputation
}

impl<'a> ParticleIndex<'a> {
    pub fn new(pg: &PixelGrid, n_particles: usize) -> Self {
        ParticleIndex {
            ak2start: vec![0; pg.size()],
            ak2end: vec![0; pg.size()],
            start2neighbors: vec![0; n_particles],
            nbrs: vec![[&[]; 27]; pg.size()]
        }
    }
    pub fn update<V>(&mut self, pg: &PixelGrid, x: &Vec<V>)
        where V: Vector<f32> + Indexable
    {
        let mut pi2ak = vec![0; x.len()];
        let mut pi2ak_sorted = vec![0; x.len()];
        self.start2neighbors = (0..pi2ak.len()).collect();
        for pi in 0..x.len() {
            pi2ak[pi] = x[pi].get_ak(pg);
        }
        // piarr:                   [0, 1, 2, 3, 4]
        // pi2ak:                   [1, 3, 5, 4, 1]
        // pi2ak_sorted:            [1, 1, 3, 4, 5]
        // start2neighbors:         [0, 4, 1, 3, 2] // start2nbrs is also just argsort of pi2ak
        // ak2start:                [N, 0, N, 2, 3, 4]
        // ak2end:                  [N, 2, N, 3, 4, 5]
        
        self.start2neighbors.sort_by(|&i, &j| pi2ak[i].cmp(&pi2ak[j]));
        for (pi, ind) in self.start2neighbors.iter().enumerate() {
            pi2ak_sorted[pi] = pi2ak[*ind];
        }
        self.ak2start = vec![pi2ak_sorted.len() + 2; pg.size()]; // out of bounds = NaN
        self.ak2end = vec![pi2ak_sorted.len() + 2; pg.size()];
        for (i, ak) in pi2ak_sorted.iter().enumerate() {
            if self.ak2start[*ak] == pi2ak_sorted.len() + 2 {
                self.ak2start[*ak] = i; // only do this if it's not already been set
            }
        }
        for (i, ak) in pi2ak_sorted.iter().enumerate().rev() {
            if self.ak2end[*ak] == pi2ak_sorted.len() + 2 {
                assert!(self.ak2start[*ak] < pi2ak_sorted.len() + 2); // must have a start
                self.ak2end[*ak] = i + 1; // only do this if it's not already been set
            }
        }
    }
    pub fn get_nbr_slices<V: Vector<f32> + Indexable>(
        &'a self, pg: &PixelGrid, v: V
    ) -> [&'a [usize]; 27] 
    {
        let mut result: [&[usize]; 27] = [&[]; 27];
        let mut idx = 0;

        // let (x, y) = pg.worldxy2xy(wx, wy);
        // let (i, j) = pg.xy2ij(x, y).unwrap();

        let (i, j, k) = v.get_ijk(pg);

        let djmin = (j as i32 - 1).max(0) as usize;
        let dimin = (i as i32 - 1).max(0) as usize;
        let dkmin = (k as i32 - 1).max(0) as usize;

        let djmax = (j as i32 + 1).min(pg.n as i32 - 1) as usize;
        let dimax = (i as i32 + 1).min(pg.m as i32 - 1) as usize;
        let dkmax = (k as i32 + 1).min(pg.l as i32 - 1) as usize;

        for dk in dkmin..=dkmax {
            for di in dimin..=dimax {
                for dj in djmin..=djmax {
                    let ak = pg.ijk2ak_nocheck(di, dj, dk);
                    let start = self.ak2start[ak];
                    if start >= self.start2neighbors.len() {
                        continue;
                    } else {
                        let end = self.ak2end[ak];
                        result[idx] = &self.start2neighbors[start..end];
                    }
                    idx += 1;
                    if idx > v.get_n_nbr_blocks() {
                        return result;
                    }
                }
            }
        }
        result
    }
    pub fn precompute_nbrs<V: Vector<f32> + Indexable>(
        &'a self, pg: &PixelGrid, x: &Vec<V>
    ) -> Vec<[&'a [usize]; 27]> {
        // you are wasting nbr computation for each cell;
        // a cell may have 10 particles. dont recompute nbrs 10 times
        // for pi, this is the nbrs for particle i
        let mut nbrs: Vec<[&'a [usize]; 27]> = vec![[&[]; 27]; pg.size()];
        let mut done: Vec<bool> = vec![false; pg.size()];
        for v in x {
            let ak = v.get_ak(pg);
            if done[ak] == false {
                nbrs[ak] = self.get_nbr_slices(&pg, *v);
                done[ak] = true;
            }
        }
        nbrs
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle_ecs::*;
    use cgmath::Vector2;

    #[test]
    fn test_particle_index_count() {
        let pg = PixelGrid::new(4, 4);
        let n_particles = pg.size();
        let mut pdata = ParticleData::<Vector2<f32>>::new(n_particles, n_particles);

        let mut pi = 0;
        for i in 0..pg.m {
            for j in 0..pg.n {
                pdata.x[pi] = Vector2::<f32>::new(j as f32, i as f32);
                pi += 1;
            }
        }
        let mut index = ParticleIndex::new(&pg, n_particles);
        index.update(&pg, &pdata.x);
        println!("");
        for i in 0..pg.m {
            for j in 0..pg.n {
                println!("{} {}: ", i, j);
                let nbrs: Vec<usize> = index.get_nbr_slices(&pg, Vector2::<f32>::new(i as f32, j as f32))
                    .iter().flat_map(|&inner| inner.iter().cloned()).collect();
                println!("{:?}", nbrs);
                assert!(nbrs.len() >= 4);
                assert!(nbrs.len() <= 9);
                for pu in nbrs {
                    print!("{} ", pu);
                }
                println!("");
            }
        }
    }
    #[test]
    fn test_precompute_isolated_particle() {
        // in this scenario, only two particles; only a few slots have them
        let pg = PixelGrid::new_with_transform(10, 10, 1.0, 1.0, -5.0, -5.0);

        let mut pdata = ParticleData::<Vector2<f32>>::new(1, 1);
        let mut index = ParticleIndex::new(&pg, 1);

        pdata.x[0] = Vector2::<f32>::new(-5.0, -5.0);
        index.update(&pg, &pdata.x);
        let pre_nbrs = index.precompute_nbrs(&pg, &pdata.x);
        assert!(pre_nbrs[0].len() > 0);
    }
    #[test]
    fn test_particle_retrieval() {
        // in this scenario, only two particles; only a few slots have them
        let pg = PixelGrid::new_with_transform(10, 10, 1.0, 1.0, -5.0, -5.0);

        let n_particles = pg.size();
        let mut pdata = ParticleData::<Vector2<f32>>::new(n_particles, n_particles);

        // arrange the particles on a grid
        for i in 0..pg.m {
            for j in 0..pg.n {
                pdata.x[i * pg.n + j] = Vector2::<f32>::new(-5.0 + j as f32 + 0.5, -5.0 + i as f32 + 0.5);
            }
        }
        let mut index = ParticleIndex::new(&pg, n_particles);
        index.update(&pg, &pdata.x);
        let mut pre_nbrs = index.precompute_nbrs(&pg, &pdata.x);

        let nbrs: Vec<usize> = index.get_nbr_slices(&pg, Vector2::<f32>::new(-5.0, -5.0))
            .iter().flat_map(|&inner| inner.iter().cloned()).collect();
        println!("{} {}: {:?} ", -5.0, -5.0, nbrs);
        assert!(nbrs.len() == 4);
        assert!(nbrs == vec![0, 1, 10, 11]);
        let (x, y) = pg.worldxy2xy(-5.0, -5.0);
        let ak = pg.xy2ak(x, y);
        let preak: Vec<usize> = pre_nbrs[ak].iter().flat_map(|&inner| inner.iter().cloned()).collect();
        println!("??{:?}", pre_nbrs);
        println!("??{:?}", pre_nbrs[ak]);
        assert!(nbrs == preak);

        let nbrs: Vec<usize> = index.get_nbr_slices(&pg, Vector2::<f32>::new(-5.0, -4.0))
            .iter().flat_map(|&inner| inner.iter().cloned()).collect();
        println!("{} {}: {:?}", -5.0, -4.0, nbrs);
        assert!(nbrs.len() == 6);
        assert!(nbrs == vec![0, 1, 10, 11, 20, 21]);

        let nbrs: Vec<usize> = index.get_nbr_slices(&pg, Vector2::<f32>::new(-4.0, -5.0))
            .iter().flat_map(|&inner| inner.iter().cloned()).collect();
        println!("{} {}: {:?}", -4.0, -5.0, nbrs);
        assert!(nbrs.len() == 6);
        assert!(nbrs == vec![0, 1, 2, 10, 11, 12]);

        let nbrs: Vec<usize> = index.get_nbr_slices(&pg, Vector2::<f32>::new(-4.0, -4.0))
            .iter().flat_map(|&inner| inner.iter().cloned()).collect();
        println!("{} {}: {:?}", -4.0, -4.0, nbrs);
        assert!(nbrs.len() == 9);
        assert!(nbrs == vec![0, 1, 2, 10, 11, 12, 20, 21, 22]);

        // now, move one of the particles to the middle
        pdata.x[0] = Vector2::<f32>::new(0.5, 0.5);
        index.update(&pg, &pdata.x);
        pre_nbrs = index.precompute_nbrs(&pg, &pdata.x);

        let nbrs: Vec<usize> = index.get_nbr_slices(&pg, Vector2::<f32>::new(0.5, 0.5))
            .iter().flat_map(|&inner| inner.iter().cloned()).collect();
        assert!(nbrs.len() == 10);
        println!("{} {}: {:?}", 0.5, 0.5, nbrs);
        assert!(nbrs.contains(&0));
        for nbrlist in &pre_nbrs {
            println!("{:?}", nbrlist);
        }

        let nbrs: Vec<usize> = index.get_nbr_slices(&pg, Vector2::<f32>::new(1.5, 1.5))
            .iter().flat_map(|&inner| inner.iter().cloned()).collect();
        assert!(nbrs.len() == 10);
        println!("{} {}: {:?}", 1.5, 1.5, nbrs);
        assert!(nbrs.contains(&0));
        let (x, y) = pg.worldxy2xy(1.5, 1.5);
        let ak = pg.xy2ak(x, y);
        println!("{}", ak);
        println!("??{:?}", pre_nbrs[ak]);
        println!("??{:?}", nbrs);
        // assert!(nbrs == pre_nbrs[ak]            
        //     .iter().flat_map(|&inner| inner.iter().cloned()).collect()
        // );


    }

}