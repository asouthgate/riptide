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
fn cal_upwind_vdqdt(fl: f32, fr: f32, q: f32, ql: f32, qr: f32, delta: f32) -> f32 {
    let cl : f32 = fl.max(0.0) * ql + fl.min(0.0) * q;
    let cr : f32 = fr.max(0.0) * q + fr.min(0.0) * qr;
    (cl - cr) / delta
} 

#[test]
fn test_cal_upwind_vqdt() {
    let result00 = cal_upwind_vdqdt(0.0, 0.0, 1.0, 1.0, 1.0, 1.0);
    assert_eq!(result00, 0.0);
    let result01 = cal_upwind_vdqdt(1.0, 0.0, 1.0, 1.0, 1.0, 1.0);
    assert_eq!(result01, 1.0);
}

/// Calculate updated velocity v' = v + dt * dvdt with only convection.
///
/// Upwinding introduces numerical diffusion, and is more stable.
/// For unrealistic simulation, this could be fine.
fn cal_new_velocity_boundary_aware_no_diffusion(
    aj: usize, dt: f32, u: &mut [f32], v: &mut [f32], boundary: &mut [f32],
    newu: &mut [f32], newv: &mut [f32], dx: f32, dy: f32, n: usize) {

    // If not boundary adjacent, this is unnecessarily slow.
    if boundary[aj] == 0.0 { // If boundary in center, bail
        newu[aj] *= boundary[aj];
        newv[aj] *= boundary[aj];
        return;
    }

    let uw = u[aj - 1];
    let uc = u[aj];
    let ue = u[aj + 1];
    let un = u[aj - n];
    let us = u[aj + n];

    let vn = v[aj - n];
    let vc = v[aj];
    let vs = v[aj + n];
    let ve = v[aj + 1];
    let vw = v[aj - 1];

    let mut flw = (uw + uc) / 2.0;
    let mut fle = (uc + ue) / 2.0;

    // Using a backstaggered grid, we need to be careful here. 
    // If 0,0 is top left, then u_ii is bottom left of v_ii
    // If 0,0 is top left, then un is v_ij + vij-1
    let mut fln = (vc + vw) / 2.0; // topl + topr = i+1j-1 + i+1j
    let mut fls = (vs + v[aj + n - 1]) / 2.0; // bl + br = ij-1 + ij

    // Enforce the boundary.
    // If we have a boundary pixel to the right, then all fluxes east are zero.
    // Likewise, boundary pixel to the left, flux west is zero.
    flw *= boundary[aj - 1];
    fle *= boundary[aj + 1];
    fln *= boundary[aj - n];
    fls *= boundary[aj + n];

    let ududx = cal_upwind_vdqdt(flw, fle, uc, uw, ue, dx);
    let vdudy = cal_upwind_vdqdt(fln, fls, uc, un, us, dy);
    let dudt = ududx + vdudy;

    // Back staggered. For v, interpolate differently.
    flw = (uc + un) / 2.0;
    fle = (ue + u[aj - n + 1]) / 2.0;
    fln = (vn + vc) / 2.0;
    fls = (vc + vs) / 2.0;

    flw *= boundary[aj - 1];
    fle *= boundary[aj + 1];
    fln *= boundary[aj - n];
    fls *= boundary[aj + n];

    let udvdx = cal_upwind_vdqdt(flw, fle, vc, vw, ve, dx);
    let vdvdy = cal_upwind_vdqdt(fln, fls, vc, vn, vs, dy);
    let dvdt = udvdx + vdvdy;

    newu[aj] = uc + ( dt * dudt );
    newv[aj] = vc + ( dt * dvdt );

    // If fluid cell is to the right of a boundary, u is zero, due to backstaggering.
    // Same situation for v.
    if boundary[aj - 1] == 0.0 {
        newu[aj] *= 0.0;
    }
    if boundary[aj - n] == 0.0 {
        newv[aj] *= 0.0;
    }
}

