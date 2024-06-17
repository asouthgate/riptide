use crate::particle_ecs::*;
use crate::pixelgrid::PixelGrid;
use crate::particle_index::*;
use crate::kernels::*;
use rayon::prelude::*;
use std::time::Instant;

use crate::vector::*;

const PI: f32 = 3.141592653589793;


/// Representation of macroscopic fluid constants for particles.
///
/// In general, these constants may either be:
///     - A single scalar for all fluid
///     - A tuple of f32
///     - A vector, one scalar for each particle type
///     - A matrix, one scalar for each pairwise combination of types
///
/// These structures are intentionally simple. We want easy conversion to GPU.
/// They could be even simpler, and represented as arrays.
pub struct ParticleConstants<V: Vector<f32>> {
    pub rho0_vec: Vec<f32>, // target density; hydrostatic force is zero at rho0
    pub c2_vec: Vec<f32>, // speed of sound squared
    pub mu_mat: Vec<Vec<f32>>, // viscosity
    pub s_mat: Vec<Vec<f32>>, // surface tension
    pub body_force: V, // e.g. gravity
    pub gamma: f32 // exponential for pressure
}


fn _debug_print_ecs<V: Vector<f32> + std::fmt::Debug>(p: ParticleRef<V>) {
    println!("x: {:?} v: {:?} a: {:?} mass: {:?} density {:?} body: {:?} drag: {:?} hydro: {:?} surface: {:?}", 
        p.x, p.v, p.a, p.mass, p.density, p.f_body, p.f_viscous, p.f_pressure, p.f_surface
    );
}


/// Calculate the hydrostatic pressure force for a given density.
///
/// This model is essentially a spring
///
/// # Arguments
///
/// * `rho` - Density
/// * `rho0` - Density at rest
/// * `k` - Either speed of sound squared, or something else
#[allow(dead_code)]
fn cal_pressure(rho: f32, rho0: f32, k: f32) -> f32 {
    k * (rho - rho0)
}

fn cal_pressure_wcsph(rho: f32, rho0: f32, c2: f32, gamma: f32) -> f32 {
    // TODO: inefficient; bweak is const
    let bweak = c2 * rho0 / gamma;
    let result = bweak * ((rho/rho0).powf(gamma) - 1.0);
    result
}

/// Calculate the jth density contribution for particle i.
#[inline]
fn cal_rho_ij(mass_j: f32, dist_ij: f32, h: f32) -> f32 {
    mass_j * debrun_spiky_kernel(dist_ij, h)
}


fn _cal_pressure_force_coefficient(pi: f32, pj: f32, rhoi: f32, rhoj: f32, mj: f32) -> f32 {
    ( (pi/rhoi.powi(2)) + (pj/rhoj.powi(2)) ) * mj
}


/// Calculate the jth pressure force contribution for particle i.
//
/// * `pi` - pressure i
/// * `pj` - pressure j
/// * `rhoi` - density i
/// * `rhoj` - density j
/// * `mj` - mass j
/// * `gradW` - gradient of the kernel function W(r, h)
fn cal_pressure_force_ij<V: Vector<f32>>(pi: f32, pj: f32, rhoi: f32, rhoj: f32, mj: f32, gradw: V) -> V {
    let pforce_coefficient = - pi * _cal_pressure_force_coefficient(pi, pj, rhoi, rhoj, mj);
    gradw * pforce_coefficient
}


pub fn update_empty_test_ecs<V: Vector<f32> + Indexable>(
    x: &Vec<V>,
    pindex: &ParticleIndex,
    pg: &PixelGrid,
) {
    x
        .par_iter()
        .enumerate()
        .for_each(|(i, xi)| {
            let ak = xi.get_ak(pg);
            for slice in &pindex.nbrs[ak] {
                for j in slice.iter() {
                    let _k = (i + *j) as f32;
                }
            }
        })
}


// Update particle densities.
//
// This function takes a vector of particles, and for a given particle
// recomputes the density based on neighboring particles.
//
/// * `particle_data`
/// * `pindex`
/// * `h` - characteristic length
pub fn update_densities_ecs<V: Vector<f32> + Indexable>(
    x: &Vec<V>,
    mass: &Vec<f32>,
    density: &mut Vec<f32>,
    pindex: &ParticleIndex,
    h: f32,
    pg: &PixelGrid,
) {
    density
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, densityi)| {
            let xi = x[i];
            assert!(!x[i][0].is_nan());
            assert!(!x[i][1].is_nan());
            *densityi = 0.0;
            let ak = xi.get_ak(pg);
            let slices = pindex.get_nbr_slices(&pg, xi);
            for slice in slices.iter() {
                for &j in *slice {
                let rij = (xi - x[j]).magnitude();
                    *densityi += cal_rho_ij(mass[j], rij, h);
                    assert!(mass[j] > 0.0);
                    assert!(!(*densityi).is_nan());
                }
            }
            assert!(*densityi > 0.0);
        })
}


// Update particle pressure forces.
//
/// * `particle_data`
/// * `pindex`
/// * `h` - characteristic length
pub fn update_forces_ecs<V: Vector<f32> + Indexable>(
    x: &Vec<V>, // todo, just pass in struct and use fields
    v: &Vec<V>,
    f_pressure: &mut Vec<V>,
    f_viscous: &mut Vec<V>,
    f_surface: &mut Vec<V>,
    f_body: &mut Vec<V>,
    pressure: &Vec<f32>, 
    density: &Vec<f32>, 
    mass: &Vec<f32>, 
    particle_type: &Vec<usize>,
    pindex: &ParticleIndex,
    h: f32,
    mu_mat: &Vec<Vec<f32>>,
    s_mat: &Vec<Vec<f32>>,
    body_force: V,
    pg: &PixelGrid,
) {

    let c_s = (3.0 * PI) / (2.0 * h);

    f_pressure
        .par_iter_mut()
        .zip(f_viscous.par_iter_mut())
        .zip(f_surface.par_iter_mut())
        .zip(f_body.par_iter_mut())
        .enumerate()
        .for_each(|(i, (((f_pressurei, f_viscousi), f_surfacei), f_bodyi))| {
            *f_bodyi = body_force * density[i];
            assert!(!x[i][0].is_nan());
            assert!(!x[i][1].is_nan());
            assert!(!f_pressurei[0].is_nan());
            assert!(!f_pressurei[1].is_nan());

            *f_pressurei *= 0.0;
            *f_viscousi *= 0.0;
            *f_surfacei *= 0.0;

            let mut f_pressure_tot = *f_pressurei * 0.0;
            let mut f_viscous_tot = *f_viscousi * 0.0;
            let mut f_surface_tot = *f_surfacei * 0.0;

            let slices = pindex.get_nbr_slices(&pg, x[i]);
            for (si, slice) in slices.iter().enumerate() {
                if si > x[i].get_n_nbr_blocks() {
                    break;
                }
                for &nbrj in *slice {
                    if nbrj == i {
                        continue;
                    }
                    let dxy: V = x[i] - x[nbrj];
                    assert!(!dxy[0].is_nan());
                    assert!(!dxy[1].is_nan());
                    assert!(dxy.magnitude() > 0.0);
                    let grad = debrun_spiky_kernel_grad_vec(dxy, h);
                    assert!(!grad[0].is_nan());
                    assert!(!grad[1].is_nan());
                    assert!(i < density.len());
                    assert!(nbrj < density.len());
                    assert!(density[i] > 0.0);
                    assert!(density[nbrj] > 0.0);
                    let fij = cal_pressure_force_ij(
                        pressure[i],
                        pressure[nbrj],
                        density[i],
                        density[nbrj],
                        mass[nbrj],
                        grad
                    );
                    f_pressure_tot += fij;
                    assert!(!fij[0].is_nan());
                    assert!(!fij[1].is_nan());
                    let r2 = dxy.dot(&dxy);
                    let r = r2.sqrt();
                    let duv = v[i] - v[nbrj];
                    let muij: f32 = mu_mat[particle_type[i]][particle_type[nbrj]];
                    let a: f32 = 4.0 * mass[nbrj] / (density[nbrj] * density[i]);
                    let b: f32 = grad.dot(&duv);
                    let c: V = dxy / r2;

                    if r < h {
                        let s = s_mat[particle_type[i]][particle_type[nbrj]];
                        let surfaceij = dxy * (s * (c_s * r).cos() / r);
                        f_surface_tot += surfaceij;
                    }
                    let viscousij = c * (muij * a * b);
                    f_viscous_tot += viscousij;
                }
            }
            *f_pressurei = f_pressure_tot;
            *f_viscousi = f_viscous_tot;
            *f_surfacei = f_surface_tot;
    })
}


pub fn update_pressures_ecs(
    pressure: &mut Vec<f32>,
    density: &Vec<f32>,
    particle_type: &Vec<usize>,
    n_particles: usize,
    rho0_vec: &Vec<f32>, 
    c2_vec: &Vec<f32>,
) {
    for k in 0..n_particles {
        let pk = particle_type[k];
        pressure[k] = cal_pressure_wcsph(density[k], rho0_vec[pk], c2_vec[pk], 7.0);
    }
}


pub fn leapfrog_update_acceleration_ecs<V: Vector<f32>>(
    pdata: &mut ParticleData<V>
) {
    for k in 0..pdata.n_particles {
        let ftot = 
            pdata.f_pressure[k] 
            + pdata.f_body[k]
            + pdata.f_surface[k]
            + pdata.f_viscous[k];
        pdata.a[k] = ftot / pdata.density[k];
    }
}


pub fn leapfrog_cal_forces_ecs<V: Vector<f32> + Indexable>(
    pg: &PixelGrid, pindex: &mut ParticleIndex,
    pdata: &ParticleData<V>,
    pdata_new: &mut ParticleData<V>,
    particle_constants: &ParticleConstants<V>,
    h: f32,
    n_threads: usize
) {

    let t0 = Instant::now();

    pindex.update(pg, &pdata_new.x); // should be new
    let nbrs = pindex.precompute_nbrs(pg, &pdata_new.x);

    let t1 = Instant::now();

    // update forces
    update_densities_ecs(
        &pdata_new.x, &pdata.mass, &mut pdata_new.density, &pindex, h, pg
    );
    let t2 = Instant::now();

    update_pressures_ecs(
        &mut pdata_new.pressure,
        &pdata_new.density,
        &pdata.particle_type,
        pdata.n_particles,
        &particle_constants.rho0_vec, 
        &particle_constants.c2_vec,
    );
    let t3 = Instant::now();

    update_forces_ecs(
        &pdata_new.x,
        &pdata_new.v,
        &mut pdata_new.f_pressure,
        &mut pdata_new.f_viscous,
        &mut pdata_new.f_surface,
        &mut pdata_new.f_body,
        &pdata_new.pressure, 
        &pdata_new.density,
        &pdata.mass,
        &pdata.particle_type,
        pindex,
        h, 
        &particle_constants.mu_mat,
        &particle_constants.s_mat,
        particle_constants.body_force,
        pg,
    );
    let t4 = Instant::now();
    update_empty_test_ecs(
        &pdata_new.x, pindex, pg
    );
    let t5 = Instant::now();
    println!("\t\t emptytest: {:?} cal forces: {:?}, pressure {:?}, density updates {:?}, pindex_update {:?}", t5-t4, t4-t3, t3-t2, t2-t1, t1-t0);

}

pub fn cal_dt(safety: f32, viscous_safety: f32, h: f32, cmax: f32, vmax: f32, mumax: f32) -> f32 {
    let a = safety * h / (vmax + cmax);
    let b = viscous_safety * h.powi(2) / mumax;
    a.min(b)
}

pub fn leapfrog_ecs<V: Vector<f32> + Indexable>(
    pg: &PixelGrid, index: &mut ParticleIndex,
    pdata: &ParticleData<V>,
    pdata_new: &mut ParticleData<V>,
    particle_constants: &ParticleConstants<V>, 
    dt: f32,
    h: f32,
    n_threads: usize
) {
    let t0 = Instant::now();

    for k in 0..pdata_new.n_particles {
        let bound = (1 - pdata_new.boundary[k]) as f32;
        pdata_new.v[k] = (pdata.v[k] + pdata.a[k] * dt / 2.0) * bound;
        pdata_new.x[k] = pdata.x[k] + pdata_new.v[k] * dt;
    }
    
    let t1 = Instant::now();

    leapfrog_cal_forces_ecs(
        pg, index, pdata, pdata_new, particle_constants, h, n_threads
    );

    let t2 = Instant::now();

    leapfrog_update_acceleration_ecs(pdata_new);

    let t3 = Instant::now();

    for k in 0..pdata_new.n_particles {
        let bound = (1 - pdata_new.boundary[k]) as f32;
        pdata_new.v[k] += (pdata_new.a[k] * dt / 2.0) * bound;
    } 
    let t4 = Instant::now();
    println!("\t v1: {:?}, acc {:?} forces {:?}, v1/2 {:?}", t4-t3, t3-t2, t2-t1, t1-t0);
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    use cgmath::{Vector2, Vector3};

    #[test]
    fn test_cal_p() {
        assert!(cal_pressure(1.0, 0.0, 1.0) == 1.0);
        assert!(cal_pressure(1.0, 0.0, 2.0) == 2.0);
    }

    #[test]
    fn test_cal_pressure_force_coeff_ij() {
        let rhoi = 1.0;
        let rhoj = 2.0;
        let pi = cal_pressure(rhoi, 0.0, 2.0); 
        let pj = cal_pressure(rhoj, 0.0, 2.0);
        let m = 1.0;
        assert!(pi == 2.0);
        assert!(pj == 4.0);
        println!("{} {}", _cal_pressure_force_coefficient(pi, pj, rhoi, rhoj, m), (2 + 1));
        assert!(_cal_pressure_force_coefficient(pi, pj, rhoi, rhoj, m) == (2.0 + 1.0));
    }

    #[test]
    fn test_2_particles_2d() {
        let h: f32 = 1.0;
        let dt: f32 = 0.001;
        let mut pdata = ParticleData::new(2, 2);
        pdata.x[0] = Vector2::<f32>::new(9.6, 10.0);
        pdata.x[1] = Vector2::<f32>::new(10.4, 10.0);
        let pg = PixelGrid::new(1000, 1000);
        let mut index = ParticleIndex::new(&pg, 2); 
        index.update(&pg, &pdata.x);
        let mut prev_err = 99999.0;

        let pc = ParticleConstants {
            rho0_vec: vec![1.0, 1.0],
            c2_vec: vec![3.4, 3.8],
            mu_mat: vec![
                vec![2.5, 0.01], 
                vec![0.01, 3.0]
            ],
            s_mat: vec![
                vec![0.0, 0.0], 
                vec![0.0, 0.0]
            ],
            body_force: Vector2::<f32>::new(0.0, 0.0),
            gamma: 7.0
        };

        let mut pdata_new = pdata.clone();
        assert!(pdata_new.density == pdata.density);

        for pi in 0..pdata.n_particles {
            println!("{}", pi);
            _debug_print_ecs(pdata.get_particle_ref(pi));   
            _debug_print_ecs(pdata_new.get_particle_ref(pi));   
        }
        println!("");

        leapfrog_cal_forces_ecs(
            &pg, &mut index,
            &pdata,
            &mut pdata_new,
            &pc, h, 2
        );
        leapfrog_update_acceleration_ecs(&mut pdata_new);

        for pi in 0..pdata.n_particles {
            println!("{}", pi);
            _debug_print_ecs(pdata.get_particle_ref(pi));   
            _debug_print_ecs(pdata_new.get_particle_ref(pi));   
        }
        mem::swap(&mut pdata, &mut pdata_new);
        println!("");

        for _ in 0..20 {
                        
            let mut new_err = 0.0;
            println!("Starting loop");

            for pi in 0..pdata.n_particles {
                println!("{}", pi);
                _debug_print_ecs(pdata.get_particle_ref(pi));   
                _debug_print_ecs(pdata_new.get_particle_ref(pi));   
            }

            leapfrog_ecs(
                &pg, &mut index,
                &pdata, &mut pdata_new,
                &pc, dt, h, 16
            );

            println!("");
            for pi in 0..pdata_new.n_particles {
                let rho0 = pc.rho0_vec[pdata_new.particle_type[pi]];
                new_err += (pdata_new.density[pi] - rho0).abs()
            }    
            println!("{} <= {}", new_err, prev_err);
            assert!(new_err <= prev_err);
            prev_err = new_err; 

            mem::swap(&mut pdata, &mut pdata_new);

        }
    }

    // #[test]
    // fn test_2_particles_3D() {
    //     let h: f32 = 1.0;
    //     let dt: f32 = 0.001;
    //     let mut pdata = ParticleData::new(2, 2);
    //     pdata.x[0] = Vector3::<f32>::new(9.6, 10.0, 10.0);
    //     pdata.x[1] = Vector3::<f32>::new(10.4, 10.0, 10.0);
    //     let pg = VoxelGrid::new(1000, 1000, 1000);
    //     let mut index = ParticleIndex::new(&pg, 2); 
    //     index.update(&pg, &pdata.x);
    //     let mut prev_err = 99999.0;

    //     let pc = ParticleConstants {
    //         rho0_vec: vec![1.0, 1.0],
    //         c2_vec: vec![3.4, 3.8],
    //         mu_mat: vec![
    //             vec![2.5, 0.01], 
    //             vec![0.01, 3.0]
    //         ],
    //         s_mat: vec![
    //             vec![0.0, 0.0], 
    //             vec![0.0, 0.0]
    //         ],
    //         body_force: Vector3::<f32>::new(0.0, 0.0, 0.0),
    //         gamma: 7.0
    //     };

    //     let mut pdata_new = pdata.clone();
    //     assert!(pdata_new.density == pdata.density);

    //     for pi in 0..pdata.n_particles {
    //         println!("{}", pi);
    //         _debug_print_ecs(pdata.get_particle_ref(pi));   
    //         _debug_print_ecs(pdata_new.get_particle_ref(pi));   
    //     }
    //     println!("");

    //     leapfrog_cal_forces_ecs(
    //         &pg, &mut index,
    //         &pdata,
    //         &mut pdata_new,
    //         &pc, h, 2
    //     );
    //     leapfrog_update_acceleration_ecs(&mut pdata_new);

    //     for pi in 0..pdata.n_particles {
    //         println!("{}", pi);
    //         _debug_print_ecs(pdata.get_particle_ref(pi));   
    //         _debug_print_ecs(pdata_new.get_particle_ref(pi));   
    //     }
    //     mem::swap(&mut pdata, &mut pdata_new);
    //     println!("");

    //     for _ in 0..20 {
                        
    //         let mut new_err = 0.0;
    //         println!("Starting loop");

    //         for pi in 0..pdata.n_particles {
    //             println!("{}", pi);
    //             _debug_print_ecs(pdata.get_particle_ref(pi));   
    //             _debug_print_ecs(pdata_new.get_particle_ref(pi));   
    //         }

    //         leapfrog_ecs(
    //             &pg, &mut index,
    //             &pdata, &mut pdata_new,
    //             &pc, dt, h, 16
    //         );

    //         println!("");
    //         for pi in 0..pdata_new.n_fluid_particles {
    //             let rho0 = pc.rho0_vec[pdata_new.particle_type[pi]];
    //             new_err += (pdata_new.density[pi] - rho0).abs()
    //         }    
    //         println!("{} <= {}", new_err, prev_err);
    //         assert!(new_err <= prev_err);
    //         prev_err = new_err; 

    //         mem::swap(&mut pdata, &mut pdata_new);

    //     }
    // }


}
