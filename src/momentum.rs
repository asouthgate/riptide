use crate::fluid_state::FluidState;
use crate::pixelgrid::PixelGrid;

/// Compute the upwind product q * F, for some quantity q and boundary flux F.
/// 
/// # Arguments
/// 
/// * `fl` - The flux on the 'left' (or more generally, the previous) face
/// * `fr` - The flux on the 'right' face
/// * `q`  - the quantity in the central cell
/// * `ql` - the quantity in the left cell
/// * `qr` - the quantity in the right cell
///
/// # Returns
/// 
/// the upwind scheme product v dq dt
#[inline]
fn cal_upwind_vdqdt(fl: f32, fr: f32, ql: f32, q: f32, qr: f32, delta: f32) -> f32 {
    let cl : f32 = fl.max(0.0) * ql + fl.min(0.0) * q;
    let cr : f32 = fr.max(0.0) * q + fr.min(0.0) * qr;
    (cl - cr) / delta
} 

/// Calculate updated velocity v' = v + dt * dvdt with only convection.
///
/// Upwinding introduces numerical diffusion, and is more stable.
#[inline]
pub fn cal_new_velocity_boundary_aware_no_diffusion(
    fs: &mut FluidState, pg: &PixelGrid, ak: usize, dt: f32
) {

    // If not boundary adjacent, this is unnecessarily slow.
    if fs.boundary[ak] == 0.0 { // If boundary in center, bail
        fs.newu[ak] *= fs.boundary[ak];
        fs.newv[ak] *= fs.boundary[ak];
        return;
    }

    let uw = fs.u[ak - 1];
    let uc = fs.u[ak];
    let ue = fs.u[ak + 1];
    let un = fs.u[ak - pg.n];
    let us = fs.u[ak + pg.n];

    // let vn = fs.v[ak - pg.n];
    let vc = fs.v[ak];
    let vn = fs.v[ak - pg.n];
    let vs = fs.v[ak + pg.n];
    let ve = fs.v[ak + 1];
    let vw = fs.v[ak - 1];

    let mut flw = (uw + uc) / 2.0;
    let mut fle = (uc + ue) / 2.0;

    // Using a backstaggered grid, we need to be careful here. 
    // If 0,0 is top left, then u_ii is bottom left of v_ii
    // If 0,0 is top left, then un is v_ij + vij-1
    let mut fln = (vc + vw) / 2.0; // topl + topr = i+1j-1 + i+1j
    let mut fls = (vs + fs.v[ak + pg.n - 1]) / 2.0; // bl + br = ij-1 + ij

    let ududx = cal_upwind_vdqdt(flw, fle, uw, uc, ue, pg.dx);
    let vdudy = cal_upwind_vdqdt(fln, fls, un, uc, us, pg.dy);
    let dudt = ududx + vdudy;

    // Back staggered. For v, interpolate differently.
    flw = (uc + un) / 2.0;
    fle = (ue + fs.u[ak - pg.n + 1]) / 2.0;
    fln = (vn + vc) / 2.0;
    fls = (vc + vs) / 2.0;

    let udvdx = cal_upwind_vdqdt(flw, fle, vw, vc, ve, pg.dx);
    let vdvdy = cal_upwind_vdqdt(fln, fls, vn, vc, vs, pg.dy);
    let dvdt = udvdx + vdvdy;

    fs.newu[ak] = uc + ( dt * dudt );
    fs.newv[ak] = vc + ( dt * dvdt );

    // If fluid cell is to the right of a boundary, u is zero, due to backstaggering.
    // Same situation for v.
    if fs.boundary[ak - 1] == 0.0 {
        fs.newu[ak] *= 0.0;
    }
    if fs.boundary[ak - pg.n] == 0.0 {
        fs.newv[ak] *= 0.0;
    }
}

/// Calculate updated quantity q' = q + dt * dqdt with only convection.
#[inline]
pub fn cal_new_q_boundary_aware_no_diffusion(
    q: &mut FluidState, fs: &FluidState, pg: &PixelGrid, ak: usize, dt: f32
) {

    // If not boundary adjacent, this is unnecessarily slow.
    if fs.boundary[ak] == 0.0 { // If boundary in center, bail
        q.u[ak] *= fs.boundary[ak];
        return;
    }

    let flw = fs.u[ak] * fs.boundary[ak];
    let fle = fs.u[ak + 1] * fs.boundary[ak + 1];
    let fln = fs.v[ak] * fs.boundary[ak];
    let fls = fs.v[ak + pg.n] * fs.boundary[ak + pg.n];
    let qc = q.u[ak];
    let qw = q.u[ak-1];
    let qe = q.u[ak+1];
    let qs = q.u[ak+pg.n];
    let qn = q.u[ak-pg.n];

    let Fy = cal_upwind_vdqdt(flw, fle, qw, qc, qe, pg.dy);
    let Fx = cal_upwind_vdqdt(fln, fls, qn, qc, qs, pg.dx);

    let dqdt = Fx + Fy;
    q.newu[ak] = qc + dqdt;
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cal_upwind_vqdt() {
        // Trivial
        let result00_111 = cal_upwind_vdqdt(0.0, 0.0, 1.0, 1.0, 1.0, 1.0);
        assert_eq!(result00_111, 0.0);
        let result10_111 = cal_upwind_vdqdt(1.0, 0.0, 1.0, 1.0, 1.0, 1.0);
        assert_eq!(result10_111, 1.0);
        let result01_111 = cal_upwind_vdqdt(0.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        assert_eq!(result01_111, -1.0);
        let result01_000 = cal_upwind_vdqdt(0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        assert_eq!(result01_000, 0.0);
        let result01_001 = cal_upwind_vdqdt(0.0, 1.0, 0.0, 0.0, 1.0, 1.0);
        assert_eq!(result01_001, 0.0);

        // Slightly less trivial
        let result10_100 = cal_upwind_vdqdt(1.0, 0.0, 1.0, 0.0, 0.0, 1.0);
        assert_eq!(result10_100, 1.0);
        let result11_00m1 = cal_upwind_vdqdt(1.0, 1.0, 0.0, 0.0, -1.0, 1.0);
        assert_eq!(result11_00m1, 0.0); // yes, zero, flux is 1.0 on rhs, even if qr is -1 (maybe a concentration)
        let result11_m111 = cal_upwind_vdqdt(1.0, 1.0, -1.0, 1.0, 1.0, 1.0);
        assert_eq!(result11_m111, -2.0); // yes, -1.0 is transported from left, 1.0 is lost from right
    }

    #[test]
    fn test_cal_new_velocity_boundary_aware_no_diffusion_two_steps_right() {   
        let pg = PixelGrid::new(6, 6);  
        let mut fs = FluidState::new(pg.m, pg.n);   
        let ak = 2 * pg.n + 2;
        fs.u[ak] = 1.0;

        // print one step right
        cal_new_velocity_boundary_aware_no_diffusion(&mut fs, &pg, ak, 1.0);
        cal_new_velocity_boundary_aware_no_diffusion(&mut fs, &pg, ak + 1, 1.0);
        assert_eq!(fs.newu[ak], 0.5);
        assert_eq!(fs.newu[ak + 1], 0.5);
        fs.swap_vectors();
        cal_new_velocity_boundary_aware_no_diffusion(&mut fs, &pg, ak, 1.0);
        cal_new_velocity_boundary_aware_no_diffusion(&mut fs, &pg, ak + 1, 1.0);
        cal_new_velocity_boundary_aware_no_diffusion(&mut fs, &pg, ak + 2, 1.0);
        assert_eq!(fs.newu[ak], 0.25);
        assert_eq!(fs.newu[ak + 1], 0.625);
        assert_eq!(fs.newu[ak + 2], 0.125);
        assert_eq!(fs.newu[ak] + fs.newu[ak + 1] + fs.newu[ak + 2], 1.0); // conservative
    }

    #[test]
    fn test_cal_new_velocity_boundary_aware_no_diffusion_two_steps_left() {   
        let pg = PixelGrid::new(6, 6);  
        let mut fs = FluidState::new(pg.m, pg.n);   
        let ak = 2 * pg.n + 4;
        fs.u[ak] = -1.0;

        // print one step right
        cal_new_velocity_boundary_aware_no_diffusion(&mut fs, &pg, ak, 1.0);
        cal_new_velocity_boundary_aware_no_diffusion(&mut fs, &pg, ak - 1, 1.0);

        assert_eq!(fs.newu[ak], -0.5);
        assert_eq!(fs.newu[ak - 1], -0.5);

        fs.swap_vectors();
        cal_new_velocity_boundary_aware_no_diffusion(&mut fs, &pg, ak, 1.0);
        cal_new_velocity_boundary_aware_no_diffusion(&mut fs, &pg, ak - 1, 1.0);
        cal_new_velocity_boundary_aware_no_diffusion(&mut fs, &pg, ak - 2, 1.0);
        assert_eq!(fs.newu[ak], -0.25);
        assert_eq!(fs.newu[ak - 1], -0.625);
        assert_eq!(fs.newu[ak - 2], -0.125);

        assert_eq!(fs.newu[ak] + fs.newu[ak - 1] + fs.newu[ak - 2], -1.0); // conservative
    }

    #[test]
    fn test_cal_new_velocity_boundary_aware_no_diffusion_two_steps_down() {   

        let pg = PixelGrid::new(6, 6);  
        let mut fs = FluidState::new(pg.m, pg.n);   
        let ak = 2 * pg.n + 2;
        fs.v[ak] = 1.0;

        cal_new_velocity_boundary_aware_no_diffusion(&mut fs, &pg, ak, 1.0);
        cal_new_velocity_boundary_aware_no_diffusion(&mut fs, &pg, ak + pg.n, 1.0);
        assert_eq!(fs.newv[ak], 0.5);
        assert_eq!(fs.newv[ak + pg.n], 0.5);
        fs.swap_vectors();
        cal_new_velocity_boundary_aware_no_diffusion(&mut fs, &pg, ak, 1.0);
        cal_new_velocity_boundary_aware_no_diffusion(&mut fs, &pg, ak + pg.n, 1.0);
        cal_new_velocity_boundary_aware_no_diffusion(&mut fs, &pg, ak + 2 * pg.n, 1.0);
        assert_eq!(fs.newv[ak], 0.25);
        assert_eq!(fs.newv[ak + pg.n], 0.625);
        assert_eq!(fs.newv[ak + 2 * pg.n], 0.125);
        assert_eq!(fs.newv[ak] + fs.newv[ak + pg.n] + fs.newv[ak + 2 * pg.n], 1.0); // conservative
        
    }

    #[test]
    fn test_cal_new_velocity_boundary_aware_no_diffusion_two_steps_up() {   
        let pg = PixelGrid::new(6, 6);  
        let mut fs = FluidState::new(pg.m, pg.n);   
        let ak = 4 * pg.n + 2;
        fs.v[ak] = -1.0;

        cal_new_velocity_boundary_aware_no_diffusion(&mut fs, &pg, ak, 1.0);
        cal_new_velocity_boundary_aware_no_diffusion(&mut fs, &pg, ak - pg.n, 1.0);

        assert_eq!(fs.newv[ak], -0.5);
        assert_eq!(fs.newv[ak - pg.n], -0.5);
        fs.swap_vectors();
        cal_new_velocity_boundary_aware_no_diffusion(&mut fs, &pg, ak, 1.0);
        cal_new_velocity_boundary_aware_no_diffusion(&mut fs, &pg, ak - pg.n, 1.0);
        cal_new_velocity_boundary_aware_no_diffusion(&mut fs, &pg, ak - 2 * pg.n, 1.0);

        assert_eq!(fs.newv[ak], -0.25);
        assert_eq!(fs.newv[ak - pg.n], -0.625);
        assert_eq!(fs.newv[ak - 2 * pg.n], -0.125);
        assert_eq!(fs.newv[ak] + fs.newv[ak - pg.n] + fs.newv[ak - 2 * pg.n], -1.0); // conservative        
    }

    #[test]
    fn test_single_pixel_boundary_adjacent_evolution() {
        let pg = PixelGrid::new(5, 5);
        let mut fs = FluidState::new(pg.m, pg.n);
        let ak0 = 2 * pg.n + 3;
        fs.u[ak0] = 0.8;
        let mut ak = 0;
        pg.print_data(&fs.u);
        for _k in 0..3 {
            fs.momentum_step(&pg, 1.0);
        }
        assert!(fs.u[ak0] < 0.8);
    }

    #[test]
    fn test_q_evolution() {
        let pg = PixelGrid::new(5, 5);
        let mut fs = FluidState::new(pg.m, pg.n);
        let mut q = FluidState::new(pg.m, pg.n); // use q as a fluid state
        let ak0 = 5 * 2 + 2;
        fs.u[ak0] = 1.0;
        fs.u[ak0 + 1] = 1.0;
        fs.u[ak0 - 1] = 1.0;
        q.u[ak0] = 1.0;
        pg.print_data(&q.u);

        q.quantity_step(&fs, &pg, 1.0);
        pg.print_data(&q.newu);
        assert!(q.u[ak0 + 1] == 1.0);

    }

}
