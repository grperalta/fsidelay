#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
==============================================================================
        FEM FOR OPTIMAL CONTROL OF A LINEAR FSI MODEL WITH DELAY
==============================================================================

This Python module approximates the solution of the following optimal
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

Here, u, p and w denote the fluid velocity, fluid pressure and strucutre
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
required in the implementation. See the accompanying file <fsinumex.py>
for the implementation of the numerical examples. 

For more details, refer to the manuscript:
    'Analysis and finite element discretization for the optimal control
    of a linear fluid-structure interaction problem with delay'
    by Gilbert Peralta and Karl Kunisch, to appear in IMA Journal of
    Numerical Analysis.


Gilbert Peralta
Department of Mathematics and Computer Science
University of the Philippines Baguio
Governor Pack Road, Baguio, Philippines 2600
Email: grperalta@up.edu.ph

"""

from __future__ import division
from os import getcwd
from numpy import linalg as la
from scipy import sparse as sp
from matplotlib import pyplot as plt
from scipy.sparse import linalg as spla
import numpy as np
import warnings
import time


__author__ = "Gilbert Peralta"
__copyright__ = "Copyright 2018, Gilbert Peralta"
__credits__ = "Karl Kunisch"
__version__ = "1.0"
__maintainer__ = "The author"
__institution__ = "University of the Philippines Baguio"
__email__ = "grperalta@up.edu.ph"
__date__ = "24 October 2018"


# GLOBAL VARIABLES

# constant pi and 2pi
PI, PI2 = np.math.pi, 2*np.math.pi

# x-coordinate, y-coordinate and radius of the circular solid domain
x0, y0, r0 = 0.3, 0.6, 0.2025


def print_line():
    """
    Prints a line on the python shell.
    """
    
    print("="*78)


class Set_Mesh_Attributes:   
    """
    Generate the attributes of the mesh needed in the implementation
    of the optimal control problem. The input data is either of the
    following: mesh.npy and meshk.mpy where k = 1, 2, 3, 4, 5. The
    mesh.npy and mesh1.npy files are generated using the Matlab package
    distmesh.m (see http://persson.berkeley.edu/distmesh/). The other
    npy files are obtained from mesh1.npy by bisection and projection of
    the new nodes on the previous interface to the physical interface
    (boundary between the fluid and solid).
    
    -----------
    Attributes:
    
        *numpy.array attributes
    
        - Node                  array of coordinates for the nodes
        - NodeSolidIndex        array of indices for the nodes in the solid
        - NodeFluidIndex        array of indices for the nodes in the fluid
        - NodeBoundaryIndex     array of indices for the nodes on the outer
                                fluid boundary
        - CenterNodeFluid       array of coordinates for the barycenters of
                                the triangles in the fluid
        - Tri                   array of geometric connectivity for the
                                triangles with respect to the ordering of
                                the points in <Node> in counterclockwise
                                orientation
        - TriFluid              array of geometric connectivity for the
                                triangles in the fluid with respect to
                                the ordering of the nodes in <Node>
        - TriSolid              array of geometric connectivity for the
                                triangles in the solid with respect to
                                the ordering of the nodes in <Node>
        - TriFluidIndex         array of indices for the triangles in the
                                fluid with respect to the ordering in <Tri>
        - EdgeInterface         array of geometric connectivity for the
                                edges on discretized interface with respect
                                to the ordering in <Node>
        - IndexFluid            array of indices for fluid nodes including
                                those for the bubble entries

        *int attributes
                            
        - NumNode               number of nodes
        - NumNodeSolid          number of nodes in the solid
        - NumNodeFluid          number of nodes in the fluid
        - NumNodeInterface      number of nodes on the interface
        - NumNodeBoudnary       number of nodes on the outer fluid boundary
        - NumTri                number of triangles
        - NumTriFluid           number of triangles in the fluid
        - NumTriSolid           number of triangles in the solid
        - dof                   total number of degrees of freedom for a
                                global velocity component at a given time
                                instant
    """
    
    def __init__(self, data):
        """
        Class construction/initialization.
        """

        # get data on the meshfiles directory
        data = getcwd() + '/meshfiles/' + data

        # load mesh data
        try:
            mesh = np.load(data, encoding='latin1')[()]
        
            # set <np.array> attributes
            self.Node = mesh['Node']
            self.NodeSolidIndex = mesh['NodeSolidIndex']
            self.NodeFluidIndex = mesh['NodeFluidIndex']
            self.NodeBoundaryIndex = mesh['NodeBoundary']
            self.CenterNodeFluid = mesh['CenterNodeFluid']
            self.Tri = mesh['Tri']
            self.TriFluid = mesh['TriFluid']
            self.TriSolid = mesh['TriSolid']
            self.TriFluidIndex = mesh['TriFluidIndex']
            self.EdgeInterface = mesh['EdgeInterface']

            # set <int> attributes
            self.NumNode = self.Node.shape[0]
            self.NumNodeSolid = self.NodeSolidIndex.shape[0]
            self.NumNodeFluid = self.NodeFluidIndex.shape[0]
            self.NumNodeInterface = self.EdgeInterface.shape[0]
            self.NumNodeBoundary = self.NodeBoundaryIndex.shape[0]
            self.NumTri = self.Tri.shape[0]
            self.NumTriFluid = self.TriFluid.shape[0]
            self.NumTriSolid = self.TriSolid.shape[0]
            self.dof = self.NumNode + self.NumTriFluid

            # set IndexFluid attribute
            self.IndexFluid = np.append(self.NodeFluidIndex,
                                        range(self.NumNode, self.dof))
        except IOError:
            message = 'No mesh file exists! See docstring for details.'
            warnings.warn(message, UserWarning)


    def size(self):
        """
        Returns the length of the largest edge among all triangles
        in the mesh.
        """
        
        h = 0.0
        for elem in range(self.NumTri):
            edge1 = (self.Node[self.Tri[elem, 1], :]
                     - self.Node[self.Tri[elem, 0], :])
            edge2 = (self.Node[self.Tri[elem, 2], :]
                     - self.Node[self.Tri[elem, 1], :])
            edge3 = (self.Node[self.Tri[elem, 0], :]
                     - self.Node[self.Tri[elem, 2], :])
            h = max(h, la.norm(edge1), la.norm(edge2), la.norm(edge3))

        return h

    def get_node_interface(self):
        """
        Returns the set of indices of the nodes on the interface.
        """
        
        return set(self.NodeFluidIndex).intersection(set(self.NodeSolidIndex))

    def get_null_control_ind(self, ControlSpec='FS_domain'):
        """
        Get the indices in the control problem for which the control is
        not active depending on the control specification.

        -----------------
        Keyword argument:
            - ControlSpec   control specification (default 'FS_domain')
                            Available methods are 'FS_domain', 'F_domain'
                            and 'S_domain'.
        """
        
        if ControlSpec is 'F_domain':
            # control acting only in the fluid domain
            list_of_ind = list(set(self.NodeSolidIndex).difference(
                self.get_node_interface()))
        elif ControlSpec is 'S_domain':
            # control acting only in the solid domain
            list_of_ind = list(set(self.IndexFluid).difference(
                self.get_node_interface()))
        elif ControlSpec is 'FS_domain':
            # control acting in the whole FSI domain
            list_of_ind = []
        try:
            return list_of_ind
        except UnboundLocalError:
            message = ("Unknown control specification: Either one of the"
                       + " following are implemented: 'FS_domain',"
                       + " 'F_domain', 'S_domain'.")
            warnings.warn(message, UserWarning)
            
    def get_coo_node(self):
        """
        Returns the x and y coordinates of the nodes in the mesh.
        """
        
        return (self.Node[:, 0], self.Node[:, 1])

    def get_coo_node_solid(self):
        """
        Returns the x and y coordinates of the nodes that lies in
        the solid domain.
        """
        
        return (self.Node[self.NodeSolidIndex, 0],
                self.Node[self.NodeSolidIndex, 1])

    def get_coo_node_fluid(self):
        """
        Returns the x and y coordinates of the nodes that lies in
        the fluid domain.
        """
        
        return (self.Node[self.NodeFluidIndex, 0],
                self.Node[self.NodeFluidIndex, 1])

    def get_centroid(self):
        """
        Returns the barycenters of the triangles in the fluid.
        """

        return (self.CenterNodeFluid[:, 0], self.CenterNodeFluid[:, 1])

    def print_details(self):
        """
        Print info of the mesh data structure.
        """

        string = ('\t Number of nodes: {:d}\n'
                  + '\t Number of nodes in the fluid domain: {:d}\n'
                  + '\t Number of nodes in the solid domain: {:d}\n'
                  + '\t Number of nodes on the interface: {:d}\n'
                  + '\t Number of triangles: {:d}\n'
                  + '\t Number of triangles in the fluid domain: {:d}\n'
                  + '\t Number of triangles in the solid domain: {:d}')
        print('Triangulation Details:')
        print(string.format(self.NumNode, self.NumNodeFluid,
                            self.NumNodeSolid, self.NumNodeInterface,
                            self.NumTri, self.NumTriFluid,
                            self.NumTriSolid))

    def locate_triangles(self):
        """
        Locate the triangles that lies in the fluid domain and re-index
        them starting from 0 with increment 1 in the succeeding triangles.
        If the triangle with global index i is in the solid domain then
        loc_tri[i] = -1.
        """
    
        loc_tri = np.ones((self.NumTri,), dtype=np.int) * (-1)
        index = 0
        for i in range(self.NumTri):
            if np.max(i == self.TriFluidIndex) == True:
                loc_tri[i] = index
                index = index + 1
            
        return loc_tri

    def locate_nodes(self):
        """
        Locate the nodes that lies in the fluid domain with respect to
        the ordering of the nodes. If the node with global index i is in
        the interior of the solid domain then loc_node[i, 1] = 0.
        """

        loc_node = np.zeros((self.NumNode, 2), dtype=np.int)
        index = 0
        for i in range(self.NumNode):
            if np.max(i == self.NodeFluidIndex) == True:
                loc_node[i, :] = [i, index]
                index = index + 1

        return loc_node

    def get_area_triangles(self, list_triangles=None):
        """
        Calculates the area of the triangles in the mesh.

        -----------------
        Keyword argument:
            - list_triangles    the list of triangles with default value
                                <None> corresponding to the whole mesh
        """
        
        if list_triangles is None:
            Tri = self.Tri
        else:
            Tri = list_triangles
            
        x21 = self.Node[Tri[:, 1], 0] - self.Node[Tri[:, 0], 0]
        x32 = self.Node[Tri[:, 2], 0] - self.Node[Tri[:, 1], 0]
        x31 = self.Node[Tri[:, 2], 0] - self.Node[Tri[:, 0], 0]
        y21 = self.Node[Tri[:, 1], 1] - self.Node[Tri[:, 0], 1]
        y32 = self.Node[Tri[:, 2], 1] - self.Node[Tri[:, 1], 1]
        y31 = self.Node[Tri[:, 2], 1] - self.Node[Tri[:, 0], 1]
        area = (x21 * y31 - y21 * x31) / 2.

        return (x21, x32, x31, y21, y32, y31, area)


    def plot(self, fignumber=1):
        """
        Plot of the mesh.

        -----------------
        Keyword argument:
            - fignumber     figure number window (defaut 1)

        """

        (x, y) = self.get_coo_node()
        theta = np.linspace(0, 2*PI, 100)
        x_circle = x0 + r0 * np.cos(theta)
        y_circle = y0 + r0 * np.sin(theta)
        
        plt.figure(fignumber)

        # plot mesh
        plt.triplot(x, y, self.Tri, 'b-', lw=1.0)

        # plot exact interface
        plt.plot(x_circle, y_circle, 'r-')        
        plt.axis('equal')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()
        

class Set_Parameters:
    """
    Set the parameters for the optimal control problem.
    
    -----------
    Attributes:
        - T                 final time horizon
        - r                 delay
        - mu                delay coefficient
        - eps               strong damping coefficient for the solid
        - tau               temporal stepsize
        - gamma_f           coefficient in the fluid velocity component
                            of the cost function
        - gamma_s1          coefficient in the solid velocity component
                            of the cost function
        - gamma_s2          coefficient in the solid displacement
                            component of the cost function
        - gamma_s3          coefficient in the solid stress component
                            of the cost function
        - alpha             Tikhonov regularization parameter for the
                            control cost
        - penal_boundary    penalization parameter for the homogeneous
                            boundary condition of the fluid equation
                            (default value 1e10)
        - artificial_pen    artificial compressibility penalization
                            parameter to eliminate the pressure in the
                            linear system (default value 1e-10)
    """
    
    # initialization
    def __init__(self, T, r, mu, eps, tau, gf, gs1, gs2, gs3, a):
        """
        Class construction/initialization.
        """

        penal_boundary = 1e10
        artificial_pen = 1e-10
        
        # parameters in the state equation
        self.T = T
        self.r = r
        self.mu = mu
        self.eps = eps
        self.tau = tau

        # parameters in the cost functional
        self.gamma_f = gf
        self.gamma_s1 = gs1
        self.gamma_s2 = gs2
        self.gamma_s3 = gs3
        self.alpha = a

        # parameters in solving the linear system
        self.penal_boundary = penal_boundary
        self.artificial_pen = artificial_pen

    def print_params(self):
        """
        Display the parameters in the state equation.
        """

        print('Parameters in the state equation:')
        string = ('\t Time Horizon (T) = {:.2f}\n'
                  + '\t Delay (r) = {:.2f}\n'
                  + '\t Delay Factor (mu) = {:.2f}\n'
                  + '\t Strong Damping Coefficient (epsilon) = {:.2f}\n'
                  + '\t Penalization Coefficient for BC = {:.2e}\n'
                  + '\t Artificial Compressibility Coefficient = {:.2e}')
        print(string.format(self.T, self.r, self.mu, self.eps,
                            self.penal_boundary, self.artificial_pen))

    def print_cost_coeffs(self):
        """
        Display the coefficients of the cost functional.
        """

        print('Coefficients in the Cost Functional:')
        string = ('\t gamma_f = {:.2e}\n' + '\t gamma_s1 = {:.2e}\n'
                  + '\t gamma_s2 = {:.2e}\n' + '\t gamma_s3= {:.2e}\n'
                  + '\t alpha = {:.2e}')
        print(string.format(self.gamma_f, self.gamma_s1, self.gamma_s2,
                            self.gamma_s3, self.alpha))


class Set_Temporal_Grid(object):
    """ Set the temporal mesh.

    -----------
    Attributes:
        - NumNode       number of nodes
        - Grid          list of nodes
        - NumHistInt    number of intervals in the subdivision of the
                        history domain
    """

    def __init__(self, prm):
        """ Class construction/initialization.

        -----------------
        Keyword argument:
            - prm   set of parameters for the optimal control problem
                    (an object with attributes <T>, <tau> and <r>)
        """

        self.Grid = np.linspace(0, prm.T, int(prm.T/prm.tau + 1))
        self.NumHistInt = int(prm.r / prm.tau)
        self.NumNode = self.Grid.shape[0]


def print_dofs(mesh, tmesh, control_spec):
    """
    Prints the number of degrees of freedom for the state variables and
    the optimization routine.

    ------------------
    Keyword arguments:
        - mesh      the spatial mesh (a class <Set_Mesh_Attributes>)
        - tmesh     the temporal mesh (a class <Set_Temporal_Grid>)
    """

    len_null_control_ind = len(mesh.get_null_control_ind(control_spec))
    total_dof = 2 * mesh.dof * (tmesh.NumNode - 1)
    total_dof_solid = 2 * mesh.NumNodeSolid * (tmesh.NumNode - 1)
    total_dof_pressure = mesh.NumNodeFluid * (tmesh.NumNode - 1)
    total_dof_opt = 2 * (mesh.dof - len_null_control_ind) * (tmesh.NumNode - 1)
    print('Number of degrees of freedom:')
    string = ('\t Global velocity: {:d}\n'
              + '\t Structure displacement: {:d}\n'
              + '\t Pressure (Eliminated): {:d}\n'
              + '\t Optimization Routine: {:d}')
    print(string.format(total_dof, total_dof_solid,
                        total_dof_pressure, total_dof_opt))


def print_meshsize(mesh, prm, tmesh):
    """
    Prints the spatial and temporal meshsizes and the number of time
    steps.

    ------------------
    Keyword arguments:
        - mesh      the spatial mesh (a class <Set_Mesh_Attributes>)
        - prm       list of parameters (a class <Set_Parameters>)
        - tmesh     the temporal mesh (a class <Set_Temporal_Grid>)
    """

    print('Mesh Size:')
    string = ('\t Space: {:.4f}\n'
              + '\t Time: {:.4f} (Number of time steps: {:d})')
    print(string.format(mesh.size(), prm.tau, tmesh.NumNode - 1))


def load_mesh(level):
    """
    Load the mesh to be used in the error analysis.

    -----------------
    Keyword argument:
        - level     level of refinement of the initial mesh
                    (available options 0, 1, 2, 3, 4)
    """
    if level == 0:
        return Set_Mesh_Attributes('mesh1.npy')
    elif level == 1:
        return Set_Mesh_Attributes('mesh2.npy')
    elif level == 2:
        return Set_Mesh_Attributes('mesh3.npy')
    elif level == 3:
        return Set_Mesh_Attributes('mesh4.npy')
    elif level == 4:
        return Set_Mesh_Attributes('mesh5.npy')
    else:
        message = "Undefined level! Available levels are 0, 1, 2, 3, 4."
        warnings.warn(message, UserWarning)
    

def print_solid_domain_error():
    """
    Prints the discretization errors for the solid domain and the
    corresponding order of reduction.
    """

    for level in range(5):
        if level > 0:
            h_old = h
            residual_old = residual
        mesh = load_mesh(level)
        h = mesh.size()
        area = mesh.get_area_triangles(list_triangles=mesh.TriSolid)[6]
        residual = PI*r0**2 - sum(area)
        string = ("Refinement level: {}\n"
                  + "Spatial meshsize: {:.6e}\n"
                  + "Residual area: {:.6e}")
        print(string.format(level, h, residual))
        if level > 0:
            rr = (np.log(residual) - np.log(residual_old)) \
                 / (np.log(h) - np.log(h_old))
            print("Reduction rate: {:.6e}\n".format(rr))
        else:
            print("")
  

def print_solid_interface_distance():
    """
    Prints the distance between the solid interface and its discretization,
    and the corresponding reduction rate.
    """

    for level in range(5):
        if level > 0:
            h_old = h
            residual_old = residual
        mesh = load_mesh(level)
        h = mesh.size()
        dist = np.zeros((mesh.EdgeInterface.shape[0],), dtype=np.float)
        # calculate the distance of each edge to the interface
        for i in range(mesh.EdgeInterface.shape[0]):
            edge = mesh.EdgeInterface[i, :]
            edge_mid = (mesh.Node[edge[0]] + mesh.Node[edge[1]]) / 2
            vec = np.array([edge_mid[0] - x0, edge_mid[1] - y0])
            dist[i] = r0 - la.norm(vec)
        residual = max(dist)
        string = ("Refinement level: {} \n"
                  + "Spatial meshsize: {:.6e} \n"
                  + "Interface distance: {:.6e}")
        print(string.format(level, h, residual))
        if level > 0:
            rr = (np.log(residual) - np.log(residual_old)) \
                 / (np.log(h) - np.log(h_old))
            print("Reduction rate: {:.6e}\n".format(rr))
        else:
            print("")
            

def print_matrix_info(Mat):
    """
    Prints the sizes of the matrices in the assembly.

    ------------------
    Keyword arguments:
        - Mat       the class of matrices for the optimal control
                    problem (a class <Mat>)
    """

    print("Size of matrices:")
    string = ("\t Mass/Stiffness matrix: {}-by-{}\n"
              + "\t Discrete divergence matrix: {}-by-{}\n"
              + "\t Total matrix at each time step: {}-by-{}\n"
              + "\t Total matrix density: {:.6f}")
    print(string.format(Mat.M.shape[0], Mat.M.shape[1],
                        Mat.B.shape[0], Mat.B.shape[1],
                        Mat.A_total.shape[0], Mat.A_total.shape[1],
                        sparse_matrix_density(Mat.A_total)))


def assembly_main(mesh):
    """
    This function assembles the mass matrix, stiffness matrix and the
    matrix associated with the discrete divergence operator for the
    FSI domain. Linear Lagrange (P1) elements for the solid velocity
    while P1-bubble/P1 elements for the fluid velocity and pressure
    are used. The matrices are assembled as <scipy.sparse.coo_matrix>
    in compressed sparse column (CSC) format.

    -----------------
    Keyword argument:
        - mesh      the spatial mesh (a class <Set_Mesh_Attributes>)

    ------    
    Yields:
        - Mf        mass matrix for fluid
        - Ms        mass matrix for solid
        - Af        stiffness matrix for fluid
        - As        stiffness matrix for solid
        - B         matrix associated with the divergence operator

    Reference:
    J. Koko, Vectorized Matlab codes for the Stokes problem with P1-bubble/P1
    finite element, http://www.isina.fr/~jkoko/Codes/StokesP1BubbleP1.pdf
    """

    start = time.time()

    # locate fluid triangles
    loc_tri = mesh.locate_triangles()
    
    # locate fluid nodes
    loc_node = mesh.locate_nodes()

    # area of triangles
    (x21, x32, x31, y21, y32, y31, area) = mesh.get_area_triangles()  

    # local entries in the mass matrix for solid
    Ms_local = [1./12] * 6
    
    # local entries in the mass matrix for fluid
    Mf_local = [1./12, 1./12, 1./12, 3./20, 1./12,
                1./12, 3./20, 1./12, 3./20, 81./560]

    # pre-allocate row indices, column indices and entries associated
    # with As and Ms
    temp_const = 6*(mesh.NumTri-mesh.NumTriFluid)
    row_ind_solid = np.zeros((temp_const,), dtype=np.int)
    col_ind_solid = np.zeros((temp_const,), dtype=np.int)
    Ms_entry = np.zeros((temp_const,), dtype=np.float64)
    As_entry = np.zeros((temp_const,), dtype=np.float64)

    # pre-allocate row indices, column indices and entries associated
    # with Af and Mf
    temp_const = 10*mesh.NumTriFluid
    row_ind_fluid = np.zeros((temp_const,), dtype=np.int)
    col_ind_fluid = np.zeros((temp_const,), dtype=np.int)
    Mf_entry = np.zeros((temp_const,), dtype=np.float64)
    Af_entry = np.zeros((temp_const,), dtype=np.float64)

    # pre-allocate row indices, column indices associated with B
    temp_const = 24*mesh.NumTriFluid
    row_ind_B = np.zeros((temp_const,), dtype=np.int)
    col_ind_B = np.zeros((temp_const,), dtype=np.int)
    B_entry = np.zeros((temp_const,), dtype=np.float64)
    
    # counters for row and column indices for solid and fluid
    ctrs = 0
    ctrf = 0

    # main assembly
    for elem in range(mesh.NumTri):
        a = area[elem]
        xt = [x32[elem], -x31[elem], x21[elem]]
        yt = [-y32[elem], y31[elem], -y21[elem]]
        # assembly of the row, column and entries for Ms and As
        if loc_tri[elem] == -1:
            it = mesh.Tri[elem, :]
            row_ind_solid[6*ctrs : 6*(ctrs+1)] \
                = [it[0], it[0], it[0], it[1], it[1], it[2]]
            col_ind_solid[6*ctrs : 6*(ctrs+1)] \
                = [it[0], it[1], it[2], it[1], it[2], it[2]]
            Ms_entry[6*ctrs : 6*(ctrs+1)] \
                = list(a * np.array(Ms_local))
            As_entry[6*ctrs : 6*(ctrs+1)] \
                = list(1./(4*a) \
                       * np.array([0.5 * (xt[0]**2 + yt[0]**2),
                                   xt[0]*xt[1] + yt[0]*yt[1],
                                   xt[0]*xt[2] + yt[0]*yt[2],
                                   0.5 * (xt[1]**2 + yt[1]**2),
                                   xt[1]*xt[2] + yt[1]*yt[2],
                                   0.5 * (xt[2]**2 + yt[2]**2)]))
            ctrs = ctrs + 1
        else:
            # assembly of the row, column and entries for Mf and Ms
            it = [mesh.Tri[elem, 0], mesh.Tri[elem, 1], mesh.Tri[elem, 2],
                  mesh.NumNode + loc_tri[elem]]
            row_ind_fluid[10*ctrf : 10*(ctrf+1)] \
                = [it[0], it[0], it[0], it[0], it[1],
                   it[1], it[1], it[2], it[2], it[3]]
            col_ind_fluid[10*ctrf : 10*(ctrf+1)] \
                = [it[0], it[1], it[2], it[3], it[1],
                   it[2], it[3], it[2], it[3], it[3]]
            Mf_entry[10*ctrf : 10*(ctrf+1)] \
                = list(a * np.array(Mf_local))
            Abb = (81./100) * (xt[0]**2 + yt[0]**2 + xt[1]**2 + yt[1]**2
                              + xt[0]*xt[1] + yt[0]*yt[1])
            Af_entry[10*ctrf: 10*(ctrf+1)] \
                = list(1./(4*a) \
                        * np.array([0.5 * (xt[0]**2 + yt[0]**2),
                                    xt[0]*xt[1] + yt[0]*yt[1],
                                    xt[0]*xt[2] + yt[0]*yt[2],
                                    0, 0.5 * (xt[1]**2 + yt[1]**2),
                                    xt[1]*xt[2] + yt[1]*yt[2],
                                    0, 0.5 * (xt[2]**2 + yt[2]**2),
                                    0, 0.5 * Abb]))
            # assembly of the row, column and entries of B1f and B2f
            itr = loc_node[mesh.Tri[elem, :], 1]
            row_ind_B[24*ctrf : 24*(ctrf+1)] \
                = ([itr[0]] * 4 + [itr[1]] * 4 + [itr[2]] * 4) * 2
            col_ind_B[24*ctrf : 24*(ctrf+1)] \
                = it * 3 + list(np.array(it) + mesh.dof) * 3
            B_entry[24*ctrf : 24*(ctrf+1)] \
                = [-(1./6)*yt[0], -(1./6)*yt[1], -(1./6)*yt[2], (9./40)*yt[0],
                   -(1./6)*yt[0], -(1./6)*yt[1], -(1./6)*yt[2], (9./40)*yt[1],
                   -(1./6)*yt[0], -(1./6)*yt[1], -(1./6)*yt[2], (9./40)*yt[2],
                   -(1./6)*xt[0], -(1./6)*xt[1], -(1./6)*xt[2], (9./40)*xt[0],
                   -(1./6)*xt[0], -(1./6)*xt[1], -(1./6)*xt[2], (9./40)*xt[1],
                   -(1./6)*xt[0], -(1./6)*xt[1], -(1./6)*xt[2], (9./40)*xt[2]]
            ctrf = ctrf + 1

    # assembly of the upper diagonal parts of Ms, As, Mf and Af
    Ms = sp.coo_matrix((Ms_entry, (row_ind_solid, col_ind_solid)),
                       shape=(mesh.dof, mesh.dof)).tocsc()
    As = sp.coo_matrix((As_entry, (row_ind_solid, col_ind_solid)),
                       shape=(mesh.dof, mesh.dof)).tocsc()
    Mf = sp.coo_matrix((Mf_entry, (row_ind_fluid, col_ind_fluid)),
                       shape=(mesh.dof, mesh.dof)).tocsc()
    Af = sp.coo_matrix((Af_entry, (row_ind_fluid, col_ind_fluid)),
                       shape=(mesh.dof, mesh.dof)).tocsc()

    # assembly of the lower diagonal parts of Ms, As, Mf and Af
    Ms = Ms + Ms.T
    As = As + As.T
    Mf = Mf + Mf.T
    Af = Af + Af.T

    # assembly of B
    B = sp.coo_matrix((B_entry, (row_ind_B, col_ind_B)),
                      shape=(mesh.NumNodeFluid, 2*mesh.dof)).tocsc()

    end = time.time()
    print_line()
    print("Matrix Assembly Elapsed Time: " + "{:.8f}".format(end-start)
          + " seconds")
                                 
    return Ms, As, Mf, Af, B


def assembly_boundary(mesh):
    """
    This function assembles the mass matrix corresponding to the
    interface using one-dimensional P1 Lagrange elements. Stored as
    a <scipy.sparse.coo_matrix> in compressed sparse column (CSC) format.

    -----------------
    Keyword argument:
        - mesh      the spatial mesh (a class <Set_Mesh_Attributes>)

    -------    
    Yields:
        - M_bdy     mass matrix for the interface
    """

    NumEdgeInterface = mesh.EdgeInterface.shape[0]
    ctr = 0

    # pre-allocate the indices for row, column and entries
    row_ind = np.zeros((3*NumEdgeInterface,), dtype=np.int)
    col_ind = np.zeros((3*NumEdgeInterface,), dtype=np.int)
    entries = np.zeros((3*NumEdgeInterface,), dtype=np.float)

    # assembly of entries
    for i in range(NumEdgeInterface):
        edge_node1 = mesh.EdgeInterface[i, 0]
        edge_node2 = mesh.EdgeInterface[i, 1]
        edge_size = la.norm(mesh.Node[edge_node1, :] -
                            mesh.Node[edge_node2, :])
        row_ind[3*ctr : 3*(ctr+1)] = [edge_node1, edge_node1, edge_node2]
        col_ind[3*ctr : 3*(ctr+1)] = [edge_node1, edge_node2, edge_node2]
        entries[3*ctr : 3*(ctr+1)] = [edge_size/6] * 3
        ctr = ctr + 1

    # assembly of the matrix
    M_bdy = sp.coo_matrix((entries, (row_ind, col_ind)),
                          shape=(mesh.dof, mesh.dof)).tocsc()
    M_bdy = M_bdy + M_bdy.T

    return M_bdy


def assembly_penalization(matrix, mesh, penal_boundary):
    """
    This function modifies the matrix by adding a penalization term
    to compensate for homogeneous (no-slip) Dirichlet boundary conditions.
    
    ------------------
    Keyword arguments:
        - matrix            the input matrix
        - mesh              the spatial mesh (a class <Set_Mesh_Attribute>)
        - penal_boundary    penalization parameter
    """

    # row and column indices
    indices = np.append(mesh.NodeBoundaryIndex,
                        mesh.dof + mesh.NodeBoundaryIndex)
    ones = np.ones(2*mesh.NumNodeBoundary)
    matrix_penal = sp.coo_matrix((penal_boundary*ones, (indices, indices)),
                                 shape=(2*mesh.dof, 2*mesh.dof)).tocsc()
    return matrix + matrix_penal


def assembly_secondary(mesh, prm):
    """
    This function assembles the mass and stiffness matrices in the
    linear system and the cost functional, as well as the reduced
    total matrix with penalization on the outer fluid boundary for
    homogeneous no-slip boundary condition.
    
    ------------------
    Keyword arguments:
        - mesh   the spatial mesh (a class <Set_Mesh_Attribute>)
        - prm    list of parameters (a class <Set_Parameters>)

    -------
    Yields:
        - M                 combined fluid and solid mass matrix
        - A_eps             combined fluid and solid stiffness matrix
        - Rs                combined solid mass and stiffness matrix
        - Rs_block          block matrix for the combined matrix <Rs>
        - Rs_cost           cost matrix for solid
        - Rs_cost_block     block matrix for the cost of solid
        - Ms_block          mass matrix block for solid
        - Mf_block          mass matrix block for fluid
        - A_total           reduced and penalized total matrix
        - As                solid stiffness matrix
        - Mf                mass matrix for fluid
        - Ms                mass matrix for solid
        - M_bdy             mass matrix associated with the interface
    """

    # call assembly_main() function
    (Ms, As, Mf, Af, B) = assembly_main(mesh)

    # assembly of matrices needed in the solution of the linear
    # system and the cost functional
    M = Mf + Ms
    A_eps = Af + prm.eps * As
    Rs = Ms + As
    Rs_block = Rs[mesh.NodeSolidIndex, :][:, mesh.NodeSolidIndex]
    Rs_cost = prm.gamma_s2 * Ms + prm.gamma_s3 * As
    Rs_cost_block = Rs_cost[mesh.NodeSolidIndex, :][:, mesh.NodeSolidIndex]
    Ms_block = Ms[mesh.NodeSolidIndex, :][:, mesh.NodeSolidIndex]
    Mf_block = Mf[mesh.IndexFluid, :][:, mesh.IndexFluid]

    # total matrix reduction and penalization
    A_temp = (1./prm.tau) * M + A_eps + prm.tau * Rs
    A_temp = sp.kron(sp.identity(2), A_temp).tocsc()
    A_temp = A_temp + (1./prm.artificial_pen) * (B.T * B)
    A_total = assembly_penalization(A_temp, mesh, prm.penal_boundary)
    
    # assembly of boundary matrix
    M_bdy = assembly_boundary(mesh)
    
    return (M, A_eps, Rs, Rs_block, Rs_cost, Rs_cost_block,
            Ms_block, Mf_block, A_total, M_bdy, As, Mf, Ms, B)


class Set_Matrices:
    """
    Class of matrices to be used in the optimal control problem.

    ------------------
    Keyword arguments:
        - mesh   the spatial mesh (a class <Set_Mesh_Attribute>)
        - prm    list of parameters (a class <Set_Parameters>)

    ------
    Notes:
        For attributes refer to the output of the function
        <assembly_secondary()>. Additional attributes are the
        following:

        - A_total_solve     factorized total matrix used in solving
                            the linear systems at each time step of the
                            discrete state and adjoint equations
                            (an object with a <solve> method)
        - Rs_block_solve    factorized matrix for the combined solid
                            matrix used in solving the adjoint solid
                            displacement (an object with a <solve>
                            method)

        Factorization is obtained using the function <splu> in the
        module <scipy.sparse.linalg>, which utilizes the SuperLU package,
        a general purpose library for solving large sparse (not necessarily
        symmetric) linear systems. A column permutation for sparsity
        preservation via a minimum degree ordering on the structure of A^T + A
        <permc_spec="MMD_AT_PLUS_A"> was used as well as the following option:

            options=dict(DiagPivotThresh=1e-6, SymmetricMode=True,
                         PivotGrowth=True)

        where the specified threshold used for a diagonal entry to be
        an acceptable pivot is set to 1e-6. For more details, refer to:
        http://crd.lbl.gov/~xiaoye/SuperLU/
    """

    def __init__(self, mesh, prm):
        """
        Class initialization/construction.
        """

        (M, Ae, Rs, Rsb, Rsc, Rscb, Msb, Mfb, At, Mb, As, Mf, Ms, B) \
            = assembly_secondary(mesh, prm)

        self.M = M
        self.A_eps = Ae
        self.Rs = Rs
        self.Rs_block = Rsb
        self.Rs_cost = Rsc
        self.Rs_cost_block = Rscb
        self.Ms_block = Msb
        self.Mf_block = Mfb
        self.A_total = At
        self.M_bdy = Mb
        self.As = As
        self.Mf = Mf
        self.Ms = Ms
        self.B = B

        start = time.time()
        splu_opts = dict(DiagPivotThresh=1e-6, SymmetricMode=True,
                         PivotGrowth=True)
        self.A_total_solve \
            = sp.linalg.splu(self.A_total, permc_spec="MMD_AT_PLUS_A",
                             panel_size=3, options=splu_opts).solve
        self.Rs_block_solve \
            = sp.linalg.splu(self.Rs_block, permc_spec="MMD_AT_PLUS_A",
                             panel_size=3, options=splu_opts).solve
        end = time.time()
        print("SPLU Factorization Elapsed Time: " + "{:.8f}".format(end-start)
              + " seconds")
    

def sparse_matrix_density(sp_matrix):
    """
    Calculates the density of the sparse matrix sp_matrix, that is,
    the ratio between the number of nonzero entries and the size of the
    matrix.
    """

    nnz = len(sp.find(sp_matrix)[2])
    return nnz / (sp_matrix.shape[0] * sp_matrix.shape[1])


def plot_sparse_matrix(sp_matrix, fn=1, info=True, ms=1):
    """
    Plot the sparse matrix.

    ------------------
    Keyword arguments:
        - sp_matrix     the sparse matrix (type <scipy.sparse.coo_matrix>)
        - fn            figure number window (default 1)
        - info          boolean variable if to print the size and
                        density of the matrix (default <True>)
        - ms            markersize (default 1)
    """

    fig = plt.figure(fn)
    plt.spy(sp_matrix, markersize=ms)
    plt.xticks([])
    plt.yticks([])
    if info is True:
        density = sparse_matrix_density(sp_matrix)
        string = ("Size of Matrix : {}-by-{}\n "
                  + "Density : {:.4f}")
        plt.xlabel(string.format(sp_matrix.shape[0],
                                 sp_matrix.shape[1], density))
    plt.show()


def bubble_coeffs(mesh, list_funcs):
    """
    Calculate the coefficients for the bubble basis functions.

    -----------------
    Keyword argumets:
        - mesh          the spatial mesh (a class <Set_Mesh_Attributes>)
        - list_funcs    list of vector-valued functions
    """

    # pre-allocation
    f1_bub = np.zeros((mesh.NumTriFluid, len(list_funcs)), dtype=np.float)
    f2_bub = np.zeros((mesh.NumTriFluid, len(list_funcs)), dtype=np.float)

    # coordinates of the centroid of triangles in the fluid
    (x_bar, y_bar) = mesh.get_centroid()

    for i in range(mesh.NumTriFluid):
        x_tri_fluid = mesh.Node[mesh.TriFluid[i, :], 0]
        y_tri_fluid = mesh.Node[mesh.TriFluid[i, :], 1]
        Acoeff = np.array([[x_tri_fluid[0], y_tri_fluid[0], 1],
                           [x_tri_fluid[1], y_tri_fluid[1], 1],
                           [x_tri_fluid[2], y_tri_fluid[2], 1]])
        coef = la.solve(Acoeff, np.identity(3))
        coef = np.dot(np.array([x_bar[i], y_bar[i], 1]), coef)
        for j in range(len(list_funcs)):
            (f1_tri, f2_tri) = list_funcs[j](x_tri_fluid, y_tri_fluid)
            (f1_barval, f2_barval) = list_funcs[j](x_bar[i], y_bar[i])
            f1_bub[i, j] = f1_barval - np.dot(coef, f1_tri)
            f2_bub[i, j] = f2_barval - np.dot(coef, f2_tri)

    return f1_bub, f2_bub


class Set_Functions:
    """
    Class of functions for the optimal control problem including
    the required gradient, Laplacian and normal derivatives.
    """

    def __init__(self):
        """
        Class initialization/construction.
        """
        
        return None

    def vel_fluid(self, x, y):
        """
        Initial fluid velocity field.
        """
        
        vel_fluid_x = (1.0 - np.cos(PI2 * x)) * np.sin(PI2 * y)
        vel_fluid_y = (np.cos(PI2 * y) - 1.0) * np.sin(PI2 * x)
        
        return vel_fluid_x, vel_fluid_y

    def disp_solid(self, x, y):
        """
        Initial solid displacement field.
        """
        
        rho = ((x - x0)**2 + (y - y0)**2) / (r0**2)
        (vf_x, vf_y) = self.vel_fluid(x, y)
        disp_solid_x = vf_x * rho
        disp_solid_y = vf_y * rho
        
        return disp_solid_x, disp_solid_y

    def vel_global(self, x, y):
        """
        Initial global velocity field.
        """

        test = (x - x0)**2 + (y - y0)**2
        indicator_f = (test >= r0**2)
        (vf_x, vf_y) = self.vel_fluid(x, y)
        (ds_x, ds_y) = self.disp_solid(x, y)
        vel_global_x = vf_x * indicator_f + ds_x * (1.0 - indicator_f)
        vel_global_y = vf_y * indicator_f + ds_y * (1.0 - indicator_f)
        
        return vel_global_x, vel_global_y

    def prs(self, x, y):
        """
        Initial pressure.
        """

        return PI2 * (np.cos(PI2 * y) - np.cos(PI2 * x))

    def grad_vel_fluid(self, x, y):
        """
        Gradient of initial velocity field.
        """

        gvf1_dx = PI2 * np.sin(PI2 * x) * np.sin(PI2 * y)
        gvf1_dy = PI2 * (1.0 - np.cos(PI2 * x)) * np.cos(PI2 * y)
        gvf2_dx = PI2 * (np.cos(PI2 * y) - 1.0) * np.cos(PI2 * x)
        gvf2_dy = - gvf1_dx

        return gvf1_dx, gvf1_dy, gvf2_dx, gvf2_dy

    def grad_disp_solid(self, x, y):
        """
        Gradient of initial solid displacement field.
        """
        
        rho = ((x - x0)**2 + (y - y0)**2) / (r0**2)
        vf = self.vel_fluid(x, y)
        gvf = self.grad_vel_fluid(x, y)
        gds1_dx = gvf[0] * rho + (2.0/r0**2) * vf[0] * (x - x0)
        gds1_dy = gvf[1] * rho + (2.0/r0**2) * vf[0] * (y - y0)
        gds2_dx = gvf[2] * rho + (2.0/r0**2) * vf[1] * (x - x0)
        gds2_dy = gvf[3] * rho + (2.0/r0**2) * vf[1] * (y - y0)
        
        return gds1_dx, gds1_dy, gds2_dx, gds2_dy

    def grad_prs(self, x, y):
        """
        Gradient of initial pressure.
        """
        
        gp_dx = PI2 * PI2 * np.sin(PI2 * x)
        gp_dy = - PI2 * PI2 * np.sin(PI2 * y)

        return gp_dx, gp_dy

    def del_vel_fluid(self, x, y):
        """
        Laplacian of initial fluid velocity field.
        """

        dvf1 = PI2 * PI2 * (2*np.cos(PI2 * x) - 1.0) * np.sin(PI2 * y)
        dvf2 = - PI2 * PI2 * (2*np.cos(PI2 * y) - 1.0) * np.sin(PI2 * x)

        return dvf1, dvf2

    def del_disp_solid(self, x, y):
        """
        Laplacian of initial solid displacement field.
        """

        rho = ((x - x0)**2 + (y - y0)**2) / (r0**2)
        vf = self.vel_fluid(x, y)
        gvf = self.grad_vel_fluid(x, y)
        dvf = self.del_vel_fluid(x, y)
        dds1 = dvf[0] * rho \
                + (4./r0**2) * (gvf[0] * (x - x0) + gvf[1] * (y - y0) + vf[0])
        dds2 = dvf[1] * rho \
                + (4./r0**2) * (gvf[2] * (x - x0) + gvf[3] * (y - y0) + vf[1])

        return dds1, dds2

    def normal_der(self, x, y, grad):
        """
        Normal derivative in the direction of the unit outward vector
        to the solid domain.
        """
        
        nx = (x - x0) / r0
        ny = (y - y0) / r0
        g = grad(x, y)
        dn_x = g[0] * nx + g[1] * ny
        dn_y = g[2] * nx + g[3] * ny
        
        return dn_x, dn_y

    def time_coeff_vel_global(self, t):
        """
        Temporal coefficient function for global velocity field.
        """

        return np.cos(PI * t)

    def time_coeff_disp_solid(self, t):
        """
        Temporal coefficient function for solid displacement field.
        """

        return (1.0 + np.sin(PI * t)) / PI

    def time_coeff_history(self, t, r):
        """
        Temporal coefficient function for the history.
        """

        return self.time_coeff_vel_global(t - r)

    def der_time_coeff_vel_global(self, t):
        """
        Derivative of temporal coefficient for global velocity field.
        """

        return - PI * np.sin(PI * t)

    def time_coeff_prs(self, t):
        """
        Temporal coefficient for fluid pressure.
        """

        return np.sin(PI * t)

    def get_time_coeffs(self, t, r):
        """
        Returns the values of the functions needed in building
        the exact solution for the error analysis.
        """

        t_p = self.time_coeff_prs(t)
        t_vg = self.time_coeff_vel_global(t)
        t_ds = self.time_coeff_disp_solid(t)
        t_dr = self.time_coeff_history(t, r)
        dt_vg = self.der_time_coeff_vel_global(t)
        
        return (t_vg, t_ds, t_dr, dt_vg, t_p)
        
    
class FSI_Vars:
    """
    Class of variables for the optimal control problem.

    -----------
    Attributes:
        - vel_global_x      x-component of global velocity field
        - vel_global_y      y-component of global velocity field
        - disp_solid_x      x-component of solid displacement field
        - disp_solid_y      y-component of solid displacement field
    """

    def __init__(self, vg_x, vg_y, ds_x, ds_y):
        """
        Class construction/initialization.
        """

        self.vel_global_x = vg_x
        self.vel_global_y = vg_y
        self.disp_solid_x = ds_x
        self.disp_solid_y = ds_y

    def __neg__(self):
        """
        Negative of FSI_Vars.
        """

        return FSI_Vars(- self.vel_global_x, - self.vel_global_y,
                        - self.disp_solid_x, - self.disp_solid_y)
        

    def __add__(self, other):
        """
        FSI_Vars addition.
        """
        
        return FSI_Vars(self.vel_global_x + other.vel_global_x,
                        self.vel_global_y + other.vel_global_y,
                        self.disp_solid_x + other.disp_solid_x,
                        self.disp_solid_y + other.disp_solid_y)

    def __sub__(self, other):
        """
        FSI_Vars subtraction.
        """
        
        return FSI_Vars(self.vel_global_x - other.vel_global_x,
                        self.vel_global_y - other.vel_global_y,
                        self.disp_solid_x - other.disp_solid_x,
                        self.disp_solid_y - other.disp_solid_y)

    def __mul__(self, c):
        """
        FSI_Vars scalar multiplication.

        """
        
        return FSI_Vars(c * self.vel_global_x, c * self.vel_global_y,
                        c * self.disp_solid_x, c * self.disp_solid_y)

    def sliced(self):
        """
        Remove the first column of the arrays in FSI_Vars.
        """

        return FSI_Vars(self.vel_global_x[:, 1:], self.vel_global_y[:, 1:],
                        self.disp_solid_x[:, 1:], self.disp_solid_y[:, 1:])

    def invsliced(self):
        """
        Remove the last column of the arrays in FSI_Vars.
        """

        s = self.vel_global_x.shape[1] - 1
        
        return FSI_Vars(self.vel_global_x[:, :s], self.vel_global_y[:, :s],
                        self.disp_solid_x[:, :s], self.disp_solid_y[:, :s])

    def norm(self, Mat, tau):
        """
        L2-norm of FSI_Vars.
        """

        J_vx = np.sum(self.vel_global_x * (Mat.M * self.vel_global_x))
        J_vy = np.sum(self.vel_global_y * (Mat.M * self.vel_global_y))
        J_sx = np.sum(self.disp_solid_x * (Mat.Ms_block * self.disp_solid_x))
        J_sy = np.sum(self.disp_solid_x * (Mat.Ms_block * self.disp_solid_y))
        J = J_vx + J_vy + J_sx + J_sy
        
        return np.sqrt(tau * J)

    
def build_initial_data(mesh):
    """
    Construct the initial data for the state equation. Returns a class
    <FSI_Vars>.

    -----------------
    Keyword argument:
        - mesh      the spatial mesh (a class <Set_Mesh_Attributes>)
    """

    funcs = Set_Functions()
    (x, y) = mesh.get_coo_node()
    (xs, ys) = mesh.get_coo_node_solid()
    (vg_x, vg_y) = funcs.vel_global(x, y)
    (vgb_x, vgb_y) = bubble_coeffs(mesh, (funcs.vel_global, ))
    vg_x = np.append(vg_x, vgb_x)
    vg_y = np.append(vg_y, vgb_y)
    (ds_x, ds_y) = funcs.disp_solid(xs, ys)

    return FSI_Vars(vg_x, vg_y, ds_x / PI, ds_y / PI)


def rank1_dot(a, b):
    """
    Build the rank 1 matrix a * transpose(b).
    """

    return np.dot(a.reshape(len(a), 1), b.reshape(1, len(b)))
    

def build_desired_state(mesh, tmesh, init):
    """
    Construct the desired states for the optimal control problem.
    Returns a class <FSI_Vars>.

    -----------------
    Keyword arguments:
        - mesh      the spatial mesh (a class <Set_Mesh_Attributes>)
        - tmesh     the temporal mesh (a class <Set_Temporal_Grid>)
        - init      initial data (a class <FSI_Vars>)
    """

    f = Set_Functions()
    time_vel = f.time_coeff_vel_global(tmesh.Grid[1:])
    time_dsp = f.time_coeff_disp_solid(tmesh.Grid[1:]) * PI

    vg_x = rank1_dot(init.vel_global_x, time_vel)
    vg_y = rank1_dot(init.vel_global_y, time_vel)
    ds_x = rank1_dot(init.disp_solid_x, time_dsp)
    ds_y = rank1_dot(init.disp_solid_y, time_dsp)
    
    return FSI_Vars(vg_x, vg_y, ds_x, ds_y)


def discretize_history(mesh, tmesh, init, r):
    """
    Discretize initial history for solid velocity by time-averaging.

    ------------------
    Keyword arguments:
        - mesh      the spatial mesh (a class <Set_Mesh_Attributes>)
        - tmesh     the temporal mesh (a class <Set_Temporal_Grid>)
        - init      initial data (a class <FSI_Vars>)
        - r         time delay
    """

    f = Set_Functions()
    dr = (f.time_coeff_history(tmesh.Grid[:tmesh.NumHistInt], r)
          + f.time_coeff_history(tmesh.Grid[1:tmesh.NumHistInt+1], r)) / 2
    hist_vel_solid_x = rank1_dot(init.vel_global_x[mesh.NodeSolidIndex], dr)
    hist_vel_solid_y = rank1_dot(init.vel_global_y[mesh.NodeSolidIndex], dr)

    return hist_vel_solid_x, hist_vel_solid_y
    
    
def FSI_Residual(state, desired):
    """
    Compute the difference between the state and the desired state.
    Returns a class <FSI_Vars>.

    ------------------
    Keyword arguments:
        - state         the state variable (a class <FSI_Vars>)
        - desired       the desired state (a class <FSI_Vars>)
    """

    return state.sliced() - desired


def FSI_State_Solver(init, hist, q, Mat, mesh, tmesh, prm, rhs):
    """
    Solves the state equation. Returns a class <FSI_Vars>.

    ------------------
    Keyword arguments:
        - init      the initial data (a class <FSI_Vars>)
        - hist      parameters (a class <Set_Parameters>) 
        - q         control 
        - Mat       a class <Mat> of matrices
        - mesh      the spatial mesh (a class <Set_Mesh_Attributes>)
        - tmesh     the temporal mesh (a class <Set_Temporal_Grid>)
        - prm       parameters (a class <Set_Parameters>) 
    """
    
    start = time.time()

    # pre-allocation
    vel_x = np.zeros((mesh.dof, tmesh.NumNode), dtype=np.float)
    vel_y = np.zeros((mesh.dof, tmesh.NumNode), dtype=np.float)
    dsp_x = np.zeros((mesh.NumNodeSolid, tmesh.NumNode), dtype=np.float)
    dsp_y = np.zeros((mesh.NumNodeSolid, tmesh.NumNode), dtype=np.float)

    # input initial data
    vel_x[:, 0] = init.vel_global_x
    vel_y[:, 0] = init.vel_global_y
    dsp_x[:, 0] = init.disp_solid_x
    dsp_y[:, 0] = init.disp_solid_y

    # for adjustment of the size of vectors associated with the solid
    vec = np.zeros((mesh.dof,), dtype=np.float)
    
    for i in range(tmesh.NumNode-1):

        # construct components of the right hand side of the linear system
        if i < tmesh.NumHistInt:
            vec[mesh.NodeSolidIndex] \
                = prm.mu * Mat.Ms_block * hist[0][:, i] \
                + Mat.Rs_block * dsp_x[:, i]
            bx = Mat.M * (q[:mesh.dof, i] + (1./prm.tau) * vel_x[:, i]) - vec
            vec[mesh.NodeSolidIndex] \
                = prm.mu * Mat.Ms_block * hist[1][:, i] \
                + Mat.Rs_block * dsp_y[:, i]
            by = Mat.M * (q[mesh.dof:, i] + (1./prm.tau) * vel_y[:, i]) - vec
        else:
            vec[mesh.NodeSolidIndex] \
                = prm.mu * Mat.Ms_block \
                * vel_x[mesh.NodeSolidIndex, i - tmesh.NumHistInt] \
                + Mat.Rs_block * dsp_x[:, i]
            bx = Mat.M * (q[:mesh.dof, i] + (1./prm.tau) * vel_x[:, i]) - vec
            vec[mesh.NodeSolidIndex] \
                = prm.mu * Mat.Ms_block \
                * vel_y[mesh.NodeSolidIndex, i - tmesh.NumHistInt] \
                + Mat.Rs_block * dsp_y[:, i]
            by = Mat.M * (q[mesh.dof:, i] + (1./prm.tau) * vel_y[:, i]) - vec

        if rhs is not None:
            bx = rhs[:mesh.dof, i] + bx
            by = rhs[mesh.dof:, i] + by
            
        # homegeneous no-slip boundary condition
        bx[mesh.NodeBoundaryIndex] = 0
        by[mesh.NodeBoundaryIndex] = 0

        # solve the linear system
        U = Mat.A_total_solve(np.append(bx, by))
        
        # components of the global velocity field
        vel_x[:, i+1] = U[:mesh.dof]
        vel_y[:, i+1] = U[mesh.dof:]

        # components of the structure displacement field
        dsp_x[:, i+1] = prm.tau * vel_x[mesh.NodeSolidIndex, i+1] \
                        + dsp_x[:, i]
        dsp_y[:, i+1] = prm.tau * vel_y[mesh.NodeSolidIndex, i+1] \
                        + dsp_y[:, i]

    end = time.time()
    print("State Solver Elapsed Time: " + "{:.8f}".format(end-start)
          + " seconds")

    return FSI_Vars(vel_x, vel_y, dsp_x, dsp_y)


def FSI_Adjoint_Solver(res, Mat, mesh, tmesh, prm):
    """
    Solves the symmetrized adjoint equation. Returns a class <FSI_Vars>.

    ------------------
    Keyword arguments:
        - init      the initial data (a class <FSI_Vars>)
        - res       difference between state and desired states
                    (a class <FSI_Vars>) 
        - Mat       a class <Mat> of matrices
        - mesh      the spatial mesh (a class <Set_Mesh_Attributes>)
        - tmesh     the temporal mesh (a class <Set_Temporal_Grid>)
        - prm       parameters (a class <Set_Parameters>) 

    """

    start = time.time()

    # pre-allocation
    vel_x = np.zeros((mesh.dof, tmesh.NumNode), dtype=np.float)
    vel_y = np.zeros((mesh.dof, tmesh.NumNode), dtype=np.float)
    dsp_x = np.zeros((mesh.NumNodeSolid, tmesh.NumNode), dtype=np.float)
    dsp_y = np.zeros((mesh.NumNodeSolid, tmesh.NumNode), dtype=np.float)

    # for adjustment of the size of vectors associated with the
    # fluid and solid
    vecf = np.zeros((mesh.dof,), dtype=np.float)
    vecs = np.zeros((mesh.dof,), dtype=np.float)

    for i in range(tmesh.NumNode-2, -1, -1):

        # construct components of the right hand side of the linear system
        if i>= tmesh.NumNode - tmesh.NumHistInt - 1:
            vecf[mesh.IndexFluid] \
                = prm.gamma_f * Mat.Mf_block \
                * res.vel_global_x[mesh.IndexFluid, i]
            vecs[mesh.NodeSolidIndex] \
                = prm.gamma_s1 * Mat.Ms_block \
                * res.vel_global_x[mesh.NodeSolidIndex, i] \
                - Mat.Rs_block * dsp_x[:, i+1] \
                + prm.tau * Mat.Rs_cost_block * res.disp_solid_x[:, i]
            cx = (1./prm.tau) * Mat.M * vel_x[:, i+1] + vecf + vecs
            vecf[mesh.IndexFluid] \
                = prm.gamma_f * Mat.Mf_block \
                * res.vel_global_y[mesh.IndexFluid, i]
            vecs[mesh.NodeSolidIndex] \
                = prm.gamma_s1 * Mat.Ms_block \
                * res.vel_global_y[mesh.NodeSolidIndex, i] \
                - Mat.Rs_block * dsp_y[:, i+1] \
                + prm.tau * Mat.Rs_cost_block * res.disp_solid_y[:, i]
            cy = (1./prm.tau) * Mat.M * vel_y[:, i+1] + vecf + vecs
        else:
            vecf[mesh.IndexFluid] \
                = prm.gamma_f * Mat.Mf_block \
                * res.vel_global_x[mesh.IndexFluid, i]
            vecs[mesh.NodeSolidIndex] \
                = prm.gamma_s1 * Mat.Ms_block \
                * res.vel_global_x[mesh.NodeSolidIndex, i] \
                - Mat.Rs_block * dsp_x[:, i+1] \
                + prm.tau * Mat.Rs_cost_block * res.disp_solid_x[:, i] \
                - prm.mu * Mat.Ms_block \
                * vel_x[mesh.NodeSolidIndex, i + tmesh.NumHistInt + 1]
            cx = (1./prm.tau) * Mat.M * vel_x[:, i+1] + vecf + vecs
            vecf[mesh.IndexFluid] \
                = prm.gamma_f * Mat.Mf_block \
                * res.vel_global_y[mesh.IndexFluid, i]
            vecs[mesh.NodeSolidIndex] \
                = prm.gamma_s1 * Mat.Ms_block \
                * res.vel_global_y[mesh.NodeSolidIndex, i] \
                - Mat.Rs_block * dsp_y[:, i+1] \
                + prm.tau * Mat.Rs_cost_block * res.disp_solid_y[:, i] \
                - prm.mu * Mat.Ms_block \
                * vel_y[mesh.NodeSolidIndex, i + tmesh.NumHistInt + 1]
            cy = (1./prm.tau) * Mat.M * vel_y[:, i+1] + vecf + vecs

        # homogeneous no-slip boundary condition
        cx[mesh.NodeBoundaryIndex] = 0
        cy[mesh.NodeBoundaryIndex] = 0

        # solve linear system
        V = Mat.A_total_solve(np.append(cx, cy))

        # components of the adjoint fluid velocity field
        vel_x[:, i] = V[:mesh.dof]
        vel_y[:, i] = V[mesh.dof:]

        # components of the adjoint structure displacement field
        dsp_temp = Mat.Rs_cost_block * res.disp_solid_x[:, i]
        dsp_temp = Mat.Rs_block_solve(dsp_temp)
        dsp_x[:, i] = prm.tau * vel_x[mesh.NodeSolidIndex, i] \
                      + dsp_x[:, i+1] - prm.tau * dsp_temp
        dsp_temp = Mat.Rs_cost_block * res.disp_solid_y[:, i]
        dsp_temp = Mat.Rs_block_solve(dsp_temp)
        dsp_y[:, i] = prm.tau * vel_y[mesh.NodeSolidIndex, i] \
                      + dsp_y[:, i+1] - prm.tau * dsp_temp

    end = time.time()
    print("Adjoint Solver Elapsed Time: " + "{:.8f}".format(end-start)
          + " seconds")

    return FSI_Vars(vel_x, vel_y, dsp_x, dsp_y)


def FSI_Cost_Functional(res, q, Mat, mesh, prm):
    """
    Calculate the objective cost value.

    ------------------
    Keyword arguments:
        - res       difference between state and desired states
                    (a class <FSI_Vars>) 
        - q         control 
        - Mat       a class <Mat> of matrices
        - mesh      the spatial mesh (a class <Set_Mesh_Attributes>)
        - tmesh     the temporal mesh (a class <Set_Temporal_Grid>)
        - prm       parameters (a class <Set_Parameters>) 

    """

    # fluid kinetic energy
    J_fx = np.sum(res.vel_global_x[mesh.IndexFluid, :]
                  * (Mat.Mf_block * res.vel_global_x[mesh.IndexFluid, :]))
    J_fy = np.sum(res.vel_global_y[mesh.IndexFluid, :] \
                  * (Mat.Mf_block * res.vel_global_y[mesh.IndexFluid, :]))

    # structure kinetic energy
    J_sx = np.sum(res.vel_global_x[mesh.NodeSolidIndex, :]
                  * (Mat.Ms_block * res.vel_global_x[mesh.NodeSolidIndex, :]))
    J_sy = np.sum(res.vel_global_y[mesh.NodeSolidIndex, :]
                  * (Mat.Ms_block * res.vel_global_y[mesh.NodeSolidIndex, :]))

    # structure potential energy
    J_dx = np.sum(res.disp_solid_x * (Mat.Rs_cost_block * res.disp_solid_x))
    J_dy = np.sum(res.disp_solid_y * (Mat.Rs_cost_block * res.disp_solid_y))

    # Tikhonov regularization
    J_q = np.sum(q[:mesh.dof, :] * (Mat.M * q[:mesh.dof, :])) \
          + np.sum(q[mesh.dof:, :] * (Mat.M * q[mesh.dof:, :]))

    # cost functional
    J = prm.gamma_f * (J_fx + J_fy) + prm.gamma_s1 * (J_sx + J_sy) \
        + (J_dx + J_dy) + prm.alpha * J_q 
        
    return 0.5 * prm.tau * J


def FSI_Adjoint_to_Control(adjoint, NullControl):
    """
    Maps the adjoint to control to be used in calculating the directional
    derivative of the cost.

    ------------------
    Keyword arguments:
        - adjoint       the adjoint state (a class <FSI_Vars>)
        - NullControl   indices to nullify based on the control region
    """

    size = adjoint.vel_global_x.shape[1] 
    qx = adjoint.vel_global_x[:, :size-1]
    qy = adjoint.vel_global_y[:, :size-1]
    qx[NullControl, :] = 0.0
    qy[NullControl, :] = 0.0

    return np.r_[qx, qy]


def FSI_Cost_Derivative(control, adj_to_ctrl, alpha):
    """
    Derivative of the cost functional.

    ------------------
    Keyword arguments:
        - control       control variable
        - adj_to_ctrl   the value of that maps the adjoint to control
        - alpha         Tikhonov regularization parameter
    """

    return alpha * control + adj_to_ctrl


def FSI_Optimality_Residual(Mat, res, mesh, prm):
    """
    Calculates the norm of the derivative of the cost functional.

    ------------------
    Keyword arguments:
        - Mat               a class <Mat> of matrices used in the optimal
                            control problem
        - res               control residual
        - mesh              the spatial mesh (a class <Set_Mesh_Attributes>)
        - prm               parameters (a class <Set_Parameters>) 
    """

    Rx = np.sum(res[:mesh.dof, :] * (Mat.M * res[:mesh.dof, :]))
    Ry = np.sum(res[mesh.dof:, :] * (Mat.M * res[mesh.dof:, :]))
    
    return np.sqrt(prm.tau * (Rx + Ry))


def Barzilai_Borwein(ocp, SecondPoint, info=True, version=1):
    """
    Barzilai-Borwein version of the gradient method.

    The algorithm stops if the consecutive cost function values have
    relative error less than the pescribed tolerance or the maximum
    number of iterations is reached. 

    ------------------
    Keyword arguments:
        - ocp           a class for the optimal control problem
        - info          Prints the iteration number, cost value and relative
                        error of consecutive cost values. (default True).
        - version       Either 1, 2, or 3. Method of getting the steplength.
                        Let dc and dj be the residues of the control and the
                        derivative of the cost functional, and s be the
                        steplength. The following are implemented depending
                        on the value of version:
                        If <version==1> then
                            s = (dc,dj) / (dj,dj).
                        If <version==2> then
                            s = (dc,dc) / (dc,dj).
                        If <version==3> then
                            s = (dc,dj) / (dj,dj) if the iteration number is
                            odd and s = (dc,dc) / (dc,dj) otherwise.
                        Here, (,) represents the inner product in Rn.
                        The default value of version is set to 1.
        - SecondPoint   The second point of the gradient method. If value is
                        <None> then the second point is given by
                        x = x - g/|g| where x is the initial point and g is
                        its gradient value. If value is <'LS'> then the
                        second point is calculated via inexact line search
                        with Armijo steplenght criterion. 
    --------
    Returns:
        The list of state, control, adjoint and residual variables
        of the optimal control problem.

    ------
    Notes:
        The ocp class should have at least the following methods:
        <state_solver>
            A function that solves the state equation.
        <adjoint_solver>
            A function that solves the adjoint equation.
        <residual>
            A function that computes the difference between
            the state and the desired state.
        <cost>
            The cost functional.
        <der_cost>
            Derivative of the cost functional.
        <adjoint_to_control>
            A function that maps the adjoint to the control.
        <optimality_residual>
            A function that computes the measure on which the
            necessary condition is satisfied, that is, norm of
            gradient is small.
        <init_steplength>
            A function that calculates the denominator in the
            steepest descent steplength.
    """

    string = ("BARZILAI-BORWEIN GRADIENT METHOD:"
              + "\t Tolerance = {:.1e}\t Version {}\n")
    print(string.format(ocp.tol, version))
    
    # main algorithm
    start_time = time.time()
    for i in range(ocp.maxit):
        if i == 0:
            if info:
                print("Iteration: 1")
            state = ocp.state_solver(ocp.init_control)
            residue = ocp.residual(state)
            cost_old = ocp.cost(residue, ocp.init_control)
            adjoint = ocp.adjoint_solver(residue)
            control_old = ocp.init_control
            control = ocp.init_control \
                      - ocp.der_cost(ocp.init_control,
                                     ocp.adjoint_to_control(adjoint))
            if SecondPoint is 'LS':
                num = np.sum(control * control)
                steplength = num / (2 * ocp.init_steplength(state, control))
                control = steplength * control
                state = ocp.state_solver(control)
                residue = ocp.residual(state)
                cost = ocp.cost(residue, control)
                alpha = 1
                iters = 0
                while cost > cost_old - (1e-4) * alpha * num:
                    alpha = alpha * 0.5
                    control = alpha * control
                    state = state * alpha
                    residue = ocp.residual(state)
                    cost = ocp.cost(residue, control)
                    iters = iters + 1
                if info:
                    print("Number of Backtracking Iterations: " + str(iters))
            elif SecondPoint is None:
                state = ocp.state_solver(control)
                residue = ocp.residual(state)
                cost = ocp.cost(residue, control)
                steplength = 1.0
            try:
                cost
            except UnboundLocalError:
                message = ("Undefined option: Either of the following:"
                           + " <None> or 'LS' is implemented.")
                warnings.warn(message, UserWarning)
                break
            if info:
                string = "\t Cost Value = {:.6e}"
                print(string.format(cost))
                print("\t Steplength = {:.6e}".format(steplength))
        else:
            if info:
                print("\nIteration: {}".format(i+1))
            adjoint_old = ocp.adjoint_to_control(adjoint)
            adjoint = ocp.adjoint_solver(residue)
            control_residue = control - control_old
            adjoint_residue = ocp.adjoint_to_control(adjoint) - adjoint_old
            res_dercost = ocp.der_cost(control_residue, adjoint_residue)
            if version == 1:
                steplength = np.sum(control_residue * res_dercost) \
                             / np.sum(res_dercost * res_dercost)
            elif version == 2:
                steplength = np.sum(control_residue * control_residue) \
                             / np.sum(control_residue * res_dercost)
            elif version == 3:
                if (i % 2) == 1:
                    steplength = np.sum(control_residue * res_dercost) \
                                 / np.sum(res_dercost * res_dercost)
                else:
                    steplength = np.sum(control_residue * control_residue) \
                                 / np.sum(control_residue * res_dercost) 
            control_old = control
            control = control \
                      - steplength \
                      * ocp.der_cost(control, ocp.adjoint_to_control(adjoint))
            state = ocp.state_solver(control)
            cost_old = cost
            residue = ocp.residual(state)
            cost = ocp.cost(residue, control)
            rel_error = np.abs(cost - cost_old) / cost
            if info:
                string = ("\t Cost Value = {:.6e}"
                          + "\t Relative Error = {:.6e}")
                print(string.format(cost, rel_error))
                string = ("\t Steplength = {:.6e}"
                          + "\t Optimality Res = {:.6e}")
                res = ocp.der_cost(control, ocp.adjoint_to_control(adjoint))
                opt_res = ocp.optimality_residual(res)
                print(string.format(steplength, opt_res))
            if rel_error < ocp.tol:
                print("Optimal solution found.")
                break
            if i == ocp.maxit - 1 and rel_error > ocp.tol:
                print("BB Warning: Maximum number of iterations reached"
                      "without satisfying the tolerance.")
    end_time = time.time()
    print("\t Elapsed time is " + "{:.8f}".format(end_time-start_time)
          + " seconds.")

    return state, adjoint, control, residue


class OCP:
    """
    Class for the optimal control problem.

    -----------
    Attributes:
        - prm               parameters (a class <Set_Parameters>) 
        - mesh              the spatial mesh (a class <Set_Mesh_Attributes>)
        - tmesh             the temporal mesh (a class <Set_Temporal_Grid>)
        - Mat               a class <Mat> of matrices used in the optimal
                            control problem
        - init              the initial data (a class <FSI_Vars>)
        - desired           the desired states (a class <FSI_Vars>)
        - null_control      indices to nullify based on the control region
        - hist              discretized initial history
        - init_control      initial control (default value is a
                            <numpy.ndarray> of zeros)
        - control_spec      control specification
        - maxit             maximum number of iterations in the
                            gradient algorithm (default 1000)
        - tol               tolerance in the gradient algorithm
                            (default 1e-6)
    
    """

    def __init__(self, prm, data, control_spec, tol=1e-6):
        """
        Class construction/initialization.
        """
        
        self.prm = prm
        self.mesh = Set_Mesh_Attributes(data)
        self.tmesh = Set_Temporal_Grid(self.prm)
        self.Mat = Set_Matrices(self.mesh, self.prm)
        self.init = build_initial_data(self.mesh)
        self.desired = None
        self.null_control = self.mesh.get_null_control_ind(control_spec)
        self.hist = discretize_history(self.mesh, self.tmesh,
                                       self.init, self.prm.r)
        self.init_control = np.zeros((2*self.mesh.dof, self.tmesh.NumNode-1),
                                     dtype=np.float)
        self.control_spec = control_spec
        self.maxit = 1000
        self.tol = tol
        self.rhs = None

    def state_solver(self, control):
        """
        Solves the state equation.
        """
        
        return FSI_State_Solver(self.init, self.hist, control, self.Mat,
                                self.mesh, self.tmesh, self.prm, self.rhs)

    def residual(self, state):
        """
        Computes the residual.
        """

        return FSI_Residual(state, self.desired)

    def adjoint_solver(self, residual):
        """
        Solves the adjoint equation.
        """

        return FSI_Adjoint_Solver(residual, self.Mat,
                                  self.mesh, self.tmesh, self.prm)

    def der_cost(self, control, adj_to_ctrl):
        """
        Derivative of the cost functional.
        """

        return FSI_Cost_Derivative(control, adj_to_ctrl, self.prm.alpha)

    def cost(self, residual, control):
        """
        Calculates the cost functional.
        """

        return FSI_Cost_Functional(residual, control, self.Mat,
                                   self.mesh, self.prm)

    def adjoint_to_control(self, adjoint):
        """
        Maps the adjoint equation to control.
        """

        return FSI_Adjoint_to_Control(adjoint, self.null_control)

    def optimality_residual(self, residual):
        """
        Calculates the residual with respect to the first order
        optimality condition.
        """

        return FSI_Optimality_Residual(self.Mat, residual, self.mesh, self.prm)

    def init_steplength(self, state, control):
        """
        Calculates the denominator of the steepest descent steplength.
        """

        return FSI_Cost_Functional(state, control, self.Mat,
                                   self.mesh, self.prm)


def build_exact_data_prelim(mesh, tmesh, Mat, prm):
    """
    Preliminary function for building the exact data in the error
    analysis for the optimal control problem.
    """

    f = Set_Functions()

    # pre-allocation
    del_vf_x = np.zeros((mesh.NumNode,), dtype=np.float)
    del_vf_y = np.zeros((mesh.NumNode,), dtype=np.float)
    del_ds_x = np.zeros((mesh.dof,), dtype=np.float)
    del_ds_y = np.zeros((mesh.dof,), dtype=np.float)
    dp_x = np.zeros((mesh.NumNode,), dtype=np.float)
    dp_y = np.zeros((mesh.NumNode,), dtype=np.float)
    ds_x = np.zeros((mesh.dof,), dtype=np.float)
    ds_y = np.zeros((mesh.dof,), dtype=np.float)

    # coordinates of nodes in the mesh
    (x, y) = mesh.get_coo_node()
    (xs, ys) = mesh.get_coo_node_solid()
    (xf, yf) = mesh.get_coo_node_fluid()

     # build spatial coefficients
    (vg_x, vg_y) = f.vel_global(x, y)
    (del_vf_x[mesh.NodeFluidIndex], del_vf_y[mesh.NodeFluidIndex]) \
        = f.del_vel_fluid(xf, yf)
    (dp_x[mesh.NodeFluidIndex], dp_y[mesh.NodeFluidIndex]) \
        = f.grad_prs(xf, yf)
    (ds_x[mesh.NodeSolidIndex], ds_y[mesh.NodeSolidIndex]) \
        = f.disp_solid(xs, ys)
    (del_ds_x[mesh.NodeSolidIndex], del_ds_y[mesh.NodeSolidIndex]) \
        = f.del_disp_solid(xs, ys)
    (bub_vals_x, bub_vals_y) \
        = bubble_coeffs(mesh, (f.vel_fluid, f.del_vel_fluid, f.grad_prs))

    # append components for the bubble basis functions
    vg_x = np.append(vg_x, bub_vals_x[:, 0])
    vg_y = np.append(vg_y, bub_vals_y[:, 0])
    del_vf_x = np.append(del_vf_x, bub_vals_x[:, 1])
    del_vf_y = np.append(del_vf_y, bub_vals_y[:, 1])
    dp_x = np.append(dp_x, bub_vals_x[:, 2])
    dp_y = np.append(dp_y, bub_vals_y[:, 2])

    # coordinates of nodes on the interface
    node_interface = list(mesh.get_node_interface())
    x_bdy = mesh.Node[node_interface, 0]
    y_bdy = mesh.Node[node_interface, 1]

    # pre-allocation
    dn_vf_x = np.zeros((mesh.dof,), dtype=np.float)
    dn_vf_y = np.zeros((mesh.dof,), dtype=np.float)
    dn_ds_x = np.zeros((mesh.dof,), dtype=np.float)
    dn_ds_y = np.zeros((mesh.dof,), dtype=np.float)
    n_px = np.zeros((mesh.dof,), dtype=np.float)
    n_py = np.zeros((mesh.dof,), dtype=np.float)

    # boundary components
    (dn_vf_x[node_interface], dn_vf_y[node_interface]) \
        = f.normal_der(x_bdy, y_bdy, f.grad_vel_fluid)
    (dn_ds_x[node_interface], dn_ds_y[node_interface]) \
        = f.normal_der(x_bdy, y_bdy, f.grad_disp_solid)
    n_px[node_interface] = f.prs(x_bdy, y_bdy) * (x_bdy - x0) / r0
    n_py[node_interface] = f.prs(x_bdy, y_bdy) * (y_bdy - y0) / r0

    return (vg_x, vg_y, del_vf_x, del_vf_y,
            ds_x, ds_y, del_ds_x, del_ds_y,
            dp_x, dp_y, dn_vf_x, dn_vf_y,
            dn_ds_x, dn_ds_y, n_px, n_py)

def build_exact_data(mesh, tmesh, Mat, prm):
    """
    Build exact data to be used in the error analysis of the
    optimal control problem.
    """

    start = time.time()
    
    # call function <buil_exact_data_prelim()>
    (vg_x, vg_y, del_vf_x, del_vf_y,
     ds_x, ds_y, del_ds_x, del_ds_y,
     dp_x, dp_y, dn_vf_x, dn_vf_y,
     dn_ds_x, dn_ds_y, n_px, n_py) \
         = build_exact_data_prelim(mesh, tmesh, Mat, prm)
    
    f = Set_Functions()
    
    # time coefficient
    t_vf = f.time_coeff_vel_global(tmesh.Grid)
    t_ds = f.time_coeff_disp_solid(tmesh.Grid)

    # exact state
    exact_vg_x = rank1_dot(vg_x, t_vf)
    exact_vg_y = rank1_dot(vg_y, t_vf)
    exact_ds_x = rank1_dot(ds_x[mesh.NodeSolidIndex], t_ds)
    exact_ds_y = rank1_dot(ds_y[mesh.NodeSolidIndex], t_ds)
    state = FSI_Vars(exact_vg_x, exact_vg_y,
                     exact_ds_x, exact_ds_y)

    # numerical adjoint
    res = state.sliced() * 2
    adjoint = FSI_Adjoint_Solver(res, Mat, mesh, tmesh, prm)

    # numerical control
    qx_exact = - (1./prm.alpha) * adjoint.vel_global_x[:, :tmesh.NumNode-1]
    qy_exact = - (1./prm.alpha) * adjoint.vel_global_y[:, :tmesh.NumNode-1]

    # nullify indices depending on the control specification
    null_control = mesh.get_null_control_ind()
    qx_exact[null_control, :] = 0.0
    qy_exact[null_control, :] = 0.0

    # pre-allocation of the arrays needed in the build-up of the right
    # hand side
    fx = np.zeros((mesh.dof, tmesh.NumNode-1), dtype=np.float)
    fy = np.zeros((mesh.dof, tmesh.NumNode-1), dtype=np.float)
    gx = np.zeros((mesh.dof, tmesh.NumNode-1), dtype=np.float)
    gy = np.zeros((mesh.dof, tmesh.NumNode-1), dtype=np.float)

    # spatial coefficients for the right hand side
    fx1 = Mat.Mf * vg_x
    fx2 = Mat.Mf * del_vf_x
    fx3 = Mat.Mf * dp_x
    fy1 = Mat.Mf * vg_y
    fy2 = Mat.Mf * del_vf_y
    fy3 = Mat.Mf * dp_y
    gx1 = Mat.Ms * ds_x
    gx2 = Mat.Ms * del_ds_x
    gy1 = Mat.Ms * ds_y
    gy2 = Mat.Ms * del_ds_y
    
    # compute time coefficients
    (t_vg1, t_ds1, t_dr1, dt_vg1, t_p1) \
        = f.get_time_coeffs(tmesh.Grid[1:], prm.r)
    (t_vg2, t_ds2, t_dr2, dt_vg2, t_p2) \
        = f.get_time_coeffs(tmesh.Grid[:tmesh.NumNode-1], prm.r)

    # average time-cofficients
    a1 = 0.5 * (dt_vg1 + dt_vg2)
    a2 = 0.5 * (t_vg1 + t_vg2)
    a3 = 0.5 * (t_p1 + t_p2)
    a4 = 0.5 * (dt_vg1 + t_ds1 + prm.mu*t_dr1 + dt_vg2 + t_ds2 + prm.mu*t_dr2)
    a5 = 0.5 * (t_ds1 + prm.eps*t_vg1 + t_ds2 + prm.eps*t_vg2)

    # construct source terms on the PDE and boundary condition 
    fx[mesh.IndexFluid, :] \
        = rank1_dot(fx1[mesh.IndexFluid], a1) \
        - rank1_dot(fx2[mesh.IndexFluid], a2) \
        + rank1_dot(fx3[mesh.IndexFluid], a3)
    fy[mesh.IndexFluid, :] \
        = rank1_dot(fy1[mesh.IndexFluid], a1) \
        - rank1_dot(fy2[mesh.IndexFluid], a2) \
        + rank1_dot(fy3[mesh.IndexFluid], a3)
    gx[mesh.NodeSolidIndex, :] \
        = rank1_dot(gx1[mesh.NodeSolidIndex], a4) \
        - rank1_dot(gx2[mesh.NodeSolidIndex], a5)
    gy[mesh.NodeSolidIndex, :] \
        = rank1_dot(gy1[mesh.NodeSolidIndex], a4) \
        - rank1_dot(gy2[mesh.NodeSolidIndex], a5)

    bx1 = Mat.M_bdy * dn_ds_x
    bx2 = Mat.M_bdy * dn_vf_x
    bx3 = Mat.M_bdy * n_px
    by1 = Mat.M_bdy * dn_ds_y
    by2 = Mat.M_bdy * dn_vf_y
    by3 = Mat.M_bdy * n_py
    
    betax = rank1_dot(bx1, a5) - rank1_dot(bx2, a2) + rank1_dot(bx3, a3)
    betay = rank1_dot(by1, a5) - rank1_dot(by2, a2) + rank1_dot(by3, a3)

    rhsx = fx + gx + betax - Mat.M * qx_exact
    rhsy = fy + gy + betay - Mat.M * qy_exact
    
    end = time.time()
    print("Build Exact Data Elapsed Time: " + "{:.8f}".format(end-start)
          + " seconds")

    return (-state.sliced(), adjoint, 
            np.r_[qx_exact, qy_exact], np.r_[rhsx, rhsy])

