# fsidelay

This Python 2.7 module approximates the solution of the following optimal
control problem for a fluid-structure interaction model with delay on
the damping term of the structure. The fluid is modeled by the linearized 
incompressible Navier-Stokes equation and the solid/structure is governed
by the damped viscoelastic wave equation. The problem is given by:

    min (1/2){ gamma_f |u - ud|^2 + gamma_s1 |w_t - vd|^2
               + gamma_s2 |w - wd|^2 + gamma_s3 |nabla w - nabla wd|^2 }
        + (alpha/2) |q|^2,       over all q in L^2((0,T) times Omega_c),
    subject to the state equation
        u_t - Delta u + grad p = b_f q,     in (0,T) times Omega_f,
        div u = 0,                          in (0,T) times Omega_f,
        u = 0,                              on (0,T) times Gamma_f,
        u = w_t,                            on (0,T) times Gamma_s,
        w_tt - Delta w - eps Delta w_t
            + w + mu w_t(.-r) = b_s q,      on (0,T) times Omega_s,
        dw/dn + eps dw_t/dn - du/dn + pn
            = 0,                            on (0,T) times Gamma_s,
    equipped with the inital data
        u(0) = u0,          in Omega_f,
        w(0) = w0,          in Omega_s,
        w_t(0) = v0,        in Omega_s,
    and initial history
        w_t(s) = z0(s),     in (-r,0) times Omega_s.
        
Here, u, p and w denote the fluid velocity, fluid pressure and structure
displacement. The boundary conditions on the interface are obtained by
imposing continuity of the velocities and the normal stresses. We have a
no-slip boundary condition on the remaining part of the boundary of the
fluid domain. Also, dw/dn denotes the derivative in the direction of the
unit normal outward to the structure domain, hence inward to the fluid
domain. 

The solid domain Omega_s is the disk centered at (0.3, 0.6) with radius
0.2025 and the entire fluid-structure domain Omega is the unit square
(0, 1)^2. The control domain Omega_c is either Omega (b_f = b_s = 1),
Omega_f (b_f = 1 and b_s = 0) or Omega_s (b_f = 0 and b_s = 1).
Spatial discretization is based on finite elements. P1-bubble/P1 pair
(mini-element) is utilized for the fluid velocity and pressure, while
P1 elements for the solid displacement and velocity. For the time
discretization, an implicit Euler scheme is used.
The three standard python packages NumPy, SciPy and Matplotlib are
required in the implementation. See the accompanying file `fsinumex.py`
for the implementation of the numerical examples. 

If you find these codes useful, you can cite the manuscript as:

*G. Peralta and K. Kunisch, Analysis and finite element discretization 
for the optimal control of a linear fluid-structure interaction problem 
with delay, IMA Journal of Numerical Analysis, Vol. 40, pp. 140-206, 2020. 
https://doi.org/10.1093/imanum/dry070*
