//! FIRE and FIRE 2.0 inertial first-order steps.
//!
//! Bitzek, Koskinen, Gähler, Moseler, Gumbsch, *Structural Relaxation
//! Made Simple*, <https://doi.org/10.1103/PhysRevLett.97.170201>.
//! Guénolé, Nöhring, Vaid, Houllé, Xie, Prakash, Bitzek, *Assessment
//! and optimization of the fast inertial relaxation engine (FIRE)*,
//! <https://doi.org/10.1016/j.commatsci.2020.109584>.

use ndarray::Array1;

use crate::step::l2;

/// Remove outward components at active coordinate bounds.
/// Wall reactions contribute neither velocity nor inertial power.
pub(crate) fn project_box_tangent(
    bounds: &eindir_core::Bounds<f64>,
    x: &Array1<f64>,
    vector: &mut Array1<f64>,
) {
    for k in 0..x.len() {
        if (x[k] <= bounds.low[k] && vector[k] < 0.0) || (x[k] >= bounds.high[k] && vector[k] > 0.0)
        {
            vector[k] = 0.0;
        }
    }
}

/// Which FIRE integrator to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FireKind {
    /// Bitzek 2006 / eOn native: velocity-Verlet, mix after the MD step.
    V1,
    /// Guénolé 2020 FIRE 2.0: semi-implicit Euler, then velocity mixing.
    V2,
}

/// Session memory for one FIRE or FIRE 2.0 solve.
pub struct FireState {
    /// Integrator.
    pub kind: FireKind,
    /// Velocity.
    pub vel: Array1<f64>,
    /// Time step.
    pub dt: f64,
    /// Initial time step.
    dt_start: f64,
    /// Time-step floor.
    dt_min: f64,
    /// Time-step cap.
    pub dt_max: f64,
    /// Mixing parameter.
    pub alpha: f64,
    /// Reset value of [`Self::alpha`].
    pub alpha_start: f64,
    /// Consecutive downhill steps.
    pub n_pos: usize,
    /// Integration steps since the last cold start.
    steps: usize,
    /// Delay before growing `dt`.
    pub n_min: usize,
    /// `dt` growth factor.
    pub f_inc: f64,
    /// `dt` shrink factor.
    pub f_dec: f64,
    /// `alpha` decay.
    pub f_alpha: f64,
}

impl FireState {
    /// Bitzek / Guénolé defaults. `dt` is [`crate::Control::istep`].
    pub fn new(kind: FireKind, dim: usize, dt: f64) -> Self {
        let dt = if dt > 0.0 { dt } else { 0.1 };
        let (dt_max, dt_min, alpha_start, n_min) = match kind {
            FireKind::V1 | FireKind::V2 => ((dt * 2.5).max(dt), 1.0e-12, 0.1, 5),
        };
        Self {
            kind,
            vel: Array1::zeros(dim),
            dt,
            dt_start: dt,
            dt_min,
            dt_max,
            alpha: alpha_start,
            alpha_start,
            n_pos: 0,
            steps: 0,
            n_min,
            f_inc: 1.1,
            f_dec: 0.5,
            f_alpha: 0.99,
        }
    }

    /// Drop velocity and mixing; keep the current `dt`.
    pub fn reset(&mut self) {
        self.vel.fill(0.0);
        self.alpha = self.alpha_start;
        self.n_pos = 0;
        if matches!(self.kind, FireKind::V2) {
            self.dt = self.dt_start;
            self.steps = 0;
        }
    }
}

fn mix_velocity(state: &mut FireState, force: &Array1<f64>) {
    let fnorm = l2(force);
    let vnorm = l2(&state.vel);
    if fnorm <= 0.0 {
        return;
    }
    let scale = state.alpha * vnorm / fnorm;
    for i in 0..state.vel.len() {
        state.vel[i] = (1.0 - state.alpha) * state.vel[i] + scale * force[i];
    }
}

fn adapt(state: &mut FireState, power: f64) {
    if power > 0.0 {
        state.n_pos += 1;
        if state.n_pos > state.n_min {
            state.dt = (state.dt * state.f_inc).min(state.dt_max);
            state.alpha *= state.f_alpha;
        }
    } else {
        state.dt *= state.f_dec;
        if state.dt < state.dt_min {
            state.dt = state.dt_min;
        }
        state.vel.fill(0.0);
        state.alpha = state.alpha_start;
        state.n_pos = 0;
    }
}

/// Displacement `dx` from the current force `f = -g`.
///
/// FIRE 2.0 adapts on the current power, corrects an uphill step by half
/// a time step, then applies the semi-implicit Euler update before mixing.
/// FIRE 1.0 takes the Verlet step first; the caller mixes on the force at
/// the new point via [`fire_after_v1`].
pub fn fire_displacement(state: &mut FireState, force: &Array1<f64>) -> Array1<f64> {
    match state.kind {
        FireKind::V2 => {
            let power = crate::vecops::dot(force.view(), state.vel.view());
            let mut displacement = Array1::zeros(force.len());
            if power > 0.0 {
                state.n_pos += 1;
                if state.n_pos > state.n_min {
                    state.dt = (state.dt * state.f_inc).min(state.dt_max);
                    state.alpha *= state.f_alpha;
                }
            } else {
                state.n_pos = 0;
                if state.steps + 1 >= state.n_min {
                    state.dt = (state.dt * state.f_dec).max(state.dt_min);
                }
                displacement.scaled_add(-0.5 * state.dt, &state.vel);
                state.vel.fill(0.0);
                state.alpha = state.alpha_start;
            }
            for i in 0..state.vel.len() {
                state.vel[i] += force[i] * state.dt;
            }
            mix_velocity(state, force);
            displacement.scaled_add(state.dt, &state.vel);
            state.steps += 1;
            displacement
        }
        FireKind::V1 => {
            for i in 0..state.vel.len() {
                state.vel[i] += force[i] * state.dt;
            }
            &state.vel * state.dt
        }
    }
}

/// FIRE 1.0 mix and adapt after the MD step, using the new force.
pub fn fire_after_v1(state: &mut FireState, force: &Array1<f64>) {
    if !matches!(state.kind, FireKind::V1) {
        return;
    }
    let power = crate::vecops::dot(force.view(), state.vel.view());
    mix_velocity(state, force);
    adapt(state, power);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn v1_first_step_follows_the_force() {
        let mut state = FireState::new(FireKind::V1, 2, 0.2);
        let force = array![1.0, 0.0];
        let dx = fire_displacement(&mut state, &force);
        assert!(dx[0] > 0.0);
        assert!(dx[1].abs() < 1e-15);
    }

    #[test]
    fn uphill_power_resets_velocity() {
        let mut state = FireState::new(FireKind::V2, 2, 0.2);
        state.vel = array![1.0, 0.0];
        let force = array![-1.0, 0.0];
        let dx = fire_displacement(&mut state, &force);
        assert_eq!(state.n_pos, 0);
        assert!((state.alpha - state.alpha_start).abs() < 1e-15);
        assert!((state.dt - 0.2).abs() < 1e-15);
        assert!((dx[0] + 0.14).abs() < 1e-15);
    }

    #[test]
    fn v2_delays_the_first_shrink() {
        let mut state = FireState::new(FireKind::V2, 1, 0.1);
        state.steps = 4;
        state.vel[0] = 1.0;
        let _ = fire_displacement(&mut state, &array![-1.0]);
        assert!((state.dt - 0.05).abs() < 1e-15);
    }
}
