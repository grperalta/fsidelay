#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Accompanying Python script for <fsi.py>. The current implementation
is working for MacOS. Required packages for Python are NumPy, SciPy
and Matplotlib. To display the numerical examples in the paper,
execute the function <main>. 
"""

from __future__ import division
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.cm as cm
import numpy as np
import matplotlib
import platform
import datetime
import fsi
import os
import sys


__author__ = "Gilbert Peralta"
__email__ = "grperalta@up.edu.ph"
__institution = "University of the Philippines Baguio"
__date__ = "24 September 2018"

def optimal_control_subroutine(OCP, NextPoint='LS'):
    """
    Returns the numerical optimal control, state, adjoint state
    and residual of the control problem.

    ------------------
    Keyword arguments:
        - OCP           class for the optimal control problem
        - NextPoint     type of computing the second point in the
                        Barzilai-Borwein gradient algorithm
    """

    # print mesh details
    fsi.print_line()
    OCP.mesh.print_details()
    
    # print meshsize
    fsi.print_line()
    fsi.print_meshsize(OCP.mesh, OCP.prm, OCP.tmesh)
    
    # print matrix details
    fsi.print_line()
    fsi.print_matrix_info(OCP.Mat)
    
    # print number of degrees of freedom
    fsi.print_line()
    fsi.print_dofs(OCP.mesh, OCP.tmesh, OCP.control_spec)
    fsi.print_line()

    return fsi.Barzilai_Borwein(OCP, SecondPoint=NextPoint,
            info=True, version=3)


def data_example1(ControlSpec, FileName, title):
    """
    Function to create and save data for Example 1. To be called by
    functions <build_data_example1_FS>, <build_data_example1_F> and
    <build_data_example1_S>.

    ------------------
    Keyword arguments:
        - ControlSpec       control specification
        - FileName          string for the file name including its
                            file extension type
        - title             title to be printed in the console
    """

    fsi.print_line()
    print("\t\t OPTIMAL CONTROL OF A LINEAR FSI MODEL WITH DELAY")
    fsi.print_line()
    print(title)
    
    # set up the parameters
    prm = fsi.Set_Parameters(T=2.0, r=1.0, mu=2.0, eps=0.1, tau=0.0025,
        gf=1.0, gs1=1.0, gs2=1.0, gs3=0.01, a=1e-6)
    fsi.print_line()
    prm.print_params()
    fsi.print_line()
    prm.print_cost_coeffs()

    # file name of mesh to be loaded
    mesh_file = 'mesh.npy'

    # set up the class for the optimal control problem
    OCP = fsi.OCP(prm, mesh_file, control_spec=ControlSpec, tol=1e-6)
    OCP.desired = fsi.build_desired_state(OCP.mesh, OCP.tmesh, OCP.init)

    # solve the optimal control problem
    (state, adjoint, control, residue) = optimal_control_subroutine(OCP)

    # save data for state, adjoint, control and residue
    data = {'state': state, 'adjoint': adjoint,
            'control': control, 'residue': residue}
    np.save(FileName, data)


def build_data_example1_FS():
    """
    Function to create and save data for Example 1 with control in
    the whole fluid-structure domain.
    """

    ControlSpec = 'FS_domain'
    FilePath = os.getcwd() + '/npyfiles/'
    FileName = FilePath + 'ex1_data_fluid_structure.npy'
    title = 'Control acting in the whole fluid-structure domain'
    data_example1(ControlSpec, FileName, title)

    
def build_data_example1_F():
    """
    Function to create and save data for Example 1 with control in the
    fluid domain.
    """

    ControlSpec = 'F_domain'
    FilePath = os.getcwd() + '/npyfiles/'
    FileName = FilePath + 'ex1_data_fluid.npy'
    title = 'Control acting in the fluid domain'
    data_example1(ControlSpec, FileName, title)


def build_data_example1_S():
    """
    Function to create and save data for Example 1 with control in the
    solid domain.
    """

    ControlSpec = 'S_domain'
    FilePath = os.getcwd() + '/npyfiles/'
    FileName = FilePath + 'ex1_data_structure.npy'
    title = 'Control acting in the structure domain'
    data_example1(ControlSpec, FileName, title)


def build_data_example2():
    """
    Function to create and save data for Example 2.
    """

    fsi.print_line()
    print("\t OPTIMAL CONTROL OF FSI WITH DELAY: CONTROL DIFFERENCE")
    fsi.print_line()
    print("Control acting in the whole FSI domain.")
    it_item = 0
    data = {}
    control_names = ['control_setup1a', 'control_setup1b',
                     'control_setup2a', 'control_setup2b']

    for item in ((0.2, 1e-3), (1.0, 1e-3), (0.2, 1e-6), (1.0, 1e-6)):
        # set up the parameters
        prm = fsi.Set_Parameters(T=2.0, r=item[0], mu=2.0, eps=0.1,
            tau=0.0025, gf=1.0, gs1=1.0, gs2=1.0, gs3=0.01, a=item[1])
        fsi.print_line()
        prm.print_params()
        fsi.print_line()
        prm.print_cost_coeffs()
        # file name of mesh to be loaded
        mesh_file = 'mesh.npy'
        # set up the class for the optimal control problem
        OCP = fsi.OCP(prm, mesh_file, control_spec='FS_domain', tol=1e-6)
        OCP.desired = fsi.build_desired_state(OCP.mesh, OCP.tmesh, OCP.init)
        # save optimal control
        control = optimal_control_subroutine(OCP)[2]
        data[control_names[it_item]] = control
        it_item = it_item + 1

    FilePath = os.getcwd() + '/npyfiles/'
    FileName = FilePath + 'ex2_data'
    np.save(FileName, data)


def display(error_table, iters):
    """
    Print the current table in the temporal and spatial
    error analysis of the optimal control problem.
    """

    fsi.print_line()
    print("CURRENT ERROR TABLE")
    print("Meshsize\t Control\t\t State\t\t\t Adjoint")
    for j in range(iters+1):
        errlist = error_table[j, :]
        print("{:.6f}\t {:.6e}\t\t {:.6e}\t\t {:.6e}".format(errlist[0],
            errlist[1], errlist[2], errlist[3]))
    if iters > 0:
        print("REDUCTION RATE FROM PREVIOUS MESH")
        reduction \
            = np.log(error_table[iters-1, 1:]/error_table[iters, 1:])/np.log(2)
        print("\t\t {:.6e}\t\t {:.6e}\t\t {:.6e}".format(reduction[0],
            reduction[1], reduction[2]))


def plot_example1_control(DataName, FigName, ticksolid=True):
    """
    Plot the numerical optimal controls with the given data name.
    """

    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.unicode'] = True

    # load control data
    FileName = os.getcwd() + '/npyfiles/' + DataName
    control = np.load(FileName, encoding='latin1')[()]['control']

    # set-up mesh and parameters
    mesh = fsi.Set_Mesh_Attributes('mesh.npy')
    prm = fsi.Set_Parameters(T=2.0, r=1.0, mu=2.0, eps=0.01, tau=0.0025,
        gf=1.0, gs1=1.0, gs2=1.0, gs3=0.01, a=1e-6)
    tmesh = fsi.Set_Temporal_Grid(prm)
    q1 = control[:mesh.NumNode, -1]
    q2 = control[mesh.dof : mesh.dof+mesh.NumNode, -1]

    # create and save figure
    fig = plt.figure(figsize=(11,7))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    (x, y) = mesh.get_coo_node()
    triang = mtri.Triangulation(x, y, triangles=mesh.Tri)
    lim1 = np.abs(np.min(q1))
    lim2 = np.abs(np.min(q2))
    surf1 = ax1.plot_trisurf(triang, q1, linewidth=0.2, antialiased=True,
        cmap=cm.coolwarm, vmin=-lim1, vmax=lim1, shade=True,
        edgecolor='black', facecolors=cm.coolwarm(q1))
    surf2 = ax2.plot_trisurf(triang, q2, linewidth=0.2, antialiased=True,
        cmap=cm.coolwarm, vmin=-lim1, vmax=lim1, shade=True,
        edgecolor='black', facecolors=cm.coolwarm(q2))

    for ax in (ax1, ax2):
        ax.view_init(azim=-135, elev=35)
        ax.set_xlim3d(0, 1)
        ax.set_ylim3d(0, 1)
        ax.grid('on')
        ax.axis('on')
        if ticksolid:
            ax.set_zticks([-800, -600, -400, -200, 0, 200])
        else:
            ax.set_zticks([-100, -50, 0, 50, 100])
    FileName = os.getcwd() + '/figfiles/' + FigName
    plt.tight_layout()
    fig.savefig(FileName, dpi=300)
    

def plot_example2():
    """
    Plot the L2-norms of the difference of controls in the set-up
    of Example 2.
    """

    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.unicode'] = True

    # load control data
    File = os.getcwd() + '/npyfiles/ex2_data.npy' 
    control_setup1a = np.load(File, encoding='latin1')[()]['control_setup1a']
    control_setup1b = np.load(File, encoding='latin1')[()]['control_setup1b']
    control_setup2a = np.load(File, encoding='latin1')[()]['control_setup2a']
    control_setup2b = np.load(File, encoding='latin1')[()]['control_setup2b']

    # pre-allocate difference of controls
    control_diff_setup1 = control_setup1a - control_setup1b
    control_diff_setup2 = control_setup2a - control_setup2b

    # load mesh and set-up parameters
    mesh = fsi.Set_Mesh_Attributes('mesh.npy')
    prm = fsi.Set_Parameters(T=2.0, r=1.0, mu=2.0, eps=0.1, tau=0.0025,
                         gf=1.0, gs1=1.0, gs2=1.0, gs3=0.01, a=1e-6)
    tmesh = fsi.Set_Temporal_Grid(prm)
    Mat = fsi.Set_Matrices(mesh, prm)

    # pre-allocation
    norm2_control_diff_setup1 = np.zeros((tmesh.NumNode-1,), dtype=np.float)
    norm2_control_diff_setup2 = np.zeros((tmesh.NumNode-1,), dtype=np.float)
    ncds_1f = np.zeros((tmesh.NumNode-1,), dtype=np.float)
    ncds_1s = np.zeros((tmesh.NumNode-1,), dtype=np.float)
    ncds_2f = np.zeros((tmesh.NumNode-1,), dtype=np.float)
    ncds_2s = np.zeros((tmesh.NumNode-1,), dtype=np.float)

    # compute L2-norms of difference in controls
    for i in range(tmesh.NumNode-1):
        vecx = control_diff_setup1[:mesh.dof, i]
        vecy = control_diff_setup1[mesh.dof:, i]
        norm2_control_diff_setup1[i] = \
            np.dot(vecx, Mat.M * vecx) + np.dot(vecy, Mat.M * vecy)
        vecx = control_diff_setup1[mesh.IndexFluid, i]
        vecy = control_diff_setup1[mesh.dof + mesh.IndexFluid, i]
        ncds_1f[i] = (np.dot(vecx, Mat.Mf_block * vecx) 
            + np.dot(vecy, Mat.Mf_block * vecy))
        vecx = control_diff_setup1[mesh.NodeSolidIndex, i]
        vecy = control_diff_setup1[mesh.dof + mesh.NodeSolidIndex, i]
        ncds_1s[i] = (np.dot(vecx, Mat.Ms_block * vecx) 
            + np.dot(vecy, Mat.Ms_block * vecy))

        vecx = control_diff_setup2[:mesh.dof, i]
        vecy = control_diff_setup2[mesh.dof:, i]
        norm2_control_diff_setup2[i] = \
            np.dot(vecx, Mat.M * vecx) + np.dot(vecy, Mat.M * vecy)
        vecx = control_diff_setup2[mesh.IndexFluid, i]
        vecy = control_diff_setup2[mesh.dof + mesh.IndexFluid, i]
        ncds_2f[i] = (np.dot(vecx, Mat.Mf_block * vecx) 
            + np.dot(vecy, Mat.Mf_block * vecy))
        vecx = control_diff_setup2[mesh.NodeSolidIndex, i]
        vecy = control_diff_setup2[mesh.dof + mesh.NodeSolidIndex, i]
        ncds_2s[i] = (np.dot(vecx, Mat.Ms_block * vecx) 
            + np.dot(vecy, Mat.Ms_block * vecy))

    # create and save figure    
    fig = plt.figure(figsize=(9,6))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    time_grid = tmesh.Grid[1:len(tmesh.Grid)-1]
    ax1.plot(time_grid, norm2_control_diff_setup1[:len(time_grid)],
             linestyle='-', color='black',
             label=r'$\|q_a(t) - q_b(t)\|_{\Omega}^2$')
    ax1.plot(time_grid, ncds_1f[:len(time_grid)],
             linestyle='-.', color='blue',
             label=r'$\|q_a(t) - q_b(t)\|_{\Omega_{fh}}^2$')
    ax1.plot(time_grid, ncds_1s[:len(time_grid)],
             linestyle='--', color='red',
             label=r'$\|q_a(t) - q_b(t)\|_{\Omega_{sh}}^2$')
    ax2.plot(time_grid, norm2_control_diff_setup2[:len(time_grid)],
             linestyle='-', color='black',
             label=r'$\|q_a(t) - q_b(t)\|_{\Omega}^2$')
    ax2.plot(time_grid, ncds_2f[:len(time_grid)],
             linestyle='-.', color='blue',
             label=r'$\|q_a(t) - q_b(t)\|_{\Omega_{fh}}^2$')
    ax2.plot(time_grid, ncds_2s[:len(time_grid)],
             linestyle='--', color='red',
             label=r'$\|q_a(t) - q_b(t)\|_{\Omega_{sh}}^2$')
    ax1.set_title(r'$\alpha = 10^{-3}$', fontsize=15)
    ax2.set_title(r'$\alpha = 10^{-6}$', fontsize=15)
    ax1.set_xlim(0, 2)
    ax1.set_ylim(0, 0.2)
    ax2.set_xlim(0, 2)
    ax2.set_ylim(0, 7)
    ax1.legend(loc='best', fontsize=14)
    ax2.legend(loc='best', fontsize=14)
    plt.subplots_adjust(left=0.12, bottom=0.11, right=0.90,
        top=0.90, wspace=0.20, hspace=0.60)
    FileName = os.getcwd() + '/figfiles/ex2.eps'
    fig.savefig(FileName, format='eps', dpi=900, bbox_inches='tight')
    

def all_residuals(res, mesh, tmesh, Mat):
    """
    Compute all residuals for the fluid velocity, solid velocity,
    solid displacement and solid stress.
    """

    As_block = Mat.As[mesh.NodeSolidIndex, :][:, mesh.NodeSolidIndex]    

    # pre-allocation
    fluid_vel = np.zeros((tmesh.NumNode-1,), dtype=np.float)
    solid_vel = np.zeros((tmesh.NumNode-1,), dtype=np.float)
    solid_dsp = np.zeros((tmesh.NumNode-1,), dtype=np.float)
    solid_str = np.zeros((tmesh.NumNode-1,), dtype=np.float)

    # loop all over time steps
    for i in range(tmesh.NumNode-1):
        vecx = res.vel_global_x[mesh.IndexFluid, i]
        vecy = res.vel_global_y[mesh.IndexFluid, i]
        fluid_vel[i] = np.sqrt(np.dot(vecx, Mat.Mf_block * vecx)
            + np.dot(vecy, Mat.Mf_block * vecy))
        vecx = res.vel_global_x[mesh.NodeSolidIndex, i]
        vecy = res.vel_global_y[mesh.NodeSolidIndex, i]
        solid_vel[i] = np.sqrt(np.dot(vecx, Mat.Ms_block * vecx)
            + np.dot(vecy, Mat.Ms_block * vecy))
        vecx = res.disp_solid_x[:, i]
        vecy = res.disp_solid_y[:, i]
        solid_dsp[i] = np.sqrt(np.dot(vecx, Mat.Ms_block * vecx)
            + np.dot(vecy, Mat.Ms_block * vecy))
        solid_str[i] = np.sqrt(np.dot(vecx, As_block * vecx)
                               + np.dot(vecy, As_block * vecy))

    return (fluid_vel, solid_vel, solid_dsp, solid_str)


def plot_example1():
    """
    Plot the difference of the norms of the difference between
    the optimal states and the desired states.
    """

    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.unicode'] = True

    # load mesh and set-up parameters
    mesh = fsi.Set_Mesh_Attributes('mesh.npy')
    prm = fsi.Set_Parameters(T=2.0, r=1.0, mu=2.0, eps=0.1, tau=0.0025,
                         gf=1.0, gs1=1.0, gs2=1.0, gs3=0.01, a=1e-6)
    tmesh = fsi.Set_Temporal_Grid(prm)
    Mat = fsi.Set_Matrices(mesh, prm)

    # create and save figure
    fig = plt.figure(figsize=(10,6))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    PathName = os.getcwd() + '/npyfiles/'
    FileName1 = PathName + 'ex1_data_fluid_structure.npy'
    FileName2 = PathName + 'ex1_data_fluid.npy'
    FileName3 = PathName + 'ex1_data_structure.npy'

    # loop over the files
    for data in [FileName1, FileName2, FileName3]:
        res = np.load(data, encoding='latin1')[()]['residue']
        (fluid_vel, solid_vel, solid_dsp, solid_str) \
            = all_residuals(res, mesh, tmesh, Mat)
        time_grid = tmesh.Grid[1:len(tmesh.Grid)-1]
        if data is FileName1:
            _color = 'black'
            _linestyle = '-'
        elif data is FileName2:
            _color = 'red'
            _linestyle = '--'
        elif data is FileName3:
            _color = 'blue'
            _linestyle = '-.'
        ax1.plot(time_grid, fluid_vel[:len(fluid_vel)-1],
                 color=_color, linestyle=_linestyle)
        ax2.plot(time_grid, solid_vel[:len(solid_vel)-1],
                 color=_color, linestyle=_linestyle)
        ax3.plot(time_grid, solid_dsp[:len(solid_dsp)-1],
                 color=_color, linestyle=_linestyle)
        ax4.plot(time_grid, solid_str[:len(solid_str)-1],
                 color=_color, linestyle=_linestyle)

    for ax in (ax1, ax2, ax3, ax4):
        ax.set_xlim(0, 2)

    ax1.set_ylim(0, 0.3)
    ax2.set_ylim(0, 0.5)
    ax3.set_ylim(0, 0.1)
    ax4.set_ylim(0, 2.0)
    ax1.set_title(r'$\|u(t) - u_d(t)\|_{\Omega_{fh}}$', fontsize=15)
    ax2.set_title(r'$\|w_t(t) - v_d(t)\|_{\Omega_{fs}}$', fontsize=15)
    ax3.set_title(r'$\|w(t) - w_d(t)\|_{\Omega_{fs}}$', fontsize=15)
    ax4.set_title(r'$\|\nabla w(t) - \nabla w_d(t)\|_{\Omega_{fs}}$',
        fontsize=15)
    plt.subplots_adjust(left=0.12, bottom=0.11, right=0.90,
        top=0.9, wspace=0.20, hspace=0.40)
    PathName = os.getcwd() + '/figfiles/ex1.eps'
    fig.savefig(PathName, format='eps', dpi=900, bbox_inches='tight')


def build_data_example3_time():
    """
    Create the data for the temporal errors in Example 3. 
    """

    fsi.print_line()
    print("\t\t OPTIMAL CONTROL OF FSI WITH DELAY: TEMPORAL ERRORS")
    iters = 0
    error_table = np.zeros((6, 4), dtype=np.float)
    time_step_list = list([0.1/(2**i) for i in range(6)])

    # loop over each time steps
    for timestep in time_step_list:
        fsi.print_line()
        print("\t\t\t REFINEMENT LEVEL: {}".format(iters))
        # set up parameters
        prm = fsi.Set_Parameters(T=0.4, r=0.1, mu=0.1, eps=0.1,
            tau=timestep, gf=1.0, gs1=1.0, gs2=1.0, gs3=0.001, a=0.1)
        # set up fsi.OCP class
        OCP = fsi.OCP(prm, data='mesh5.npy',
            control_spec='FS_domain', tol=1e-6)
        # build exact desired, adjoint, control variables
        (OCP.desired, ex_adjoint, ex_control, OCP.rhs) \
            = fsi.build_exact_data(OCP.mesh, OCP.tmesh, OCP.Mat, OCP.prm)
        # gradient algorithm
        (state, adjoint, control, residue) \
            = optimal_control_subroutine(OCP, NextPoint=None)
        # error in control
        res = ex_control - control
        err_control \
            = fsi.FSI_Optimality_Residual(OCP.Mat, res, OCP.mesh, OCP.prm)
        # error in state
        res = state.sliced() + OCP.desired
        err_state = res.norm(OCP.Mat, OCP.prm.tau)
        # error in adjoint
        res = ex_adjoint.invsliced() - adjoint.invsliced()
        err_adjoint = res.norm(OCP.Mat, OCP.prm.tau)
        error_table[iters, :] \
            = [timestep, err_control, err_state, err_adjoint]
        display(error_table, iters)
        iters = iters + 1

    FileName = os.getcwd() + '/npyfiles/ex3_temporal_error.npy'
    np.save(FileName, {'error_table': error_table})


def build_data_example3_space():
    """
    Create the data for the spatial errors in Example 3. 
    """
    
    fsi.print_line()
    print("\t\t OPTIMAL CONTROL OF FSI WITH DELAY: SPATIAL ERRORS")

    # set up parameters
    prm = fsi.Set_Parameters(T=0.4, r=0.1, mu=0.1, eps=0.1,
        tau=0.0002, gf=1.0, gs1=1.0, gs2=1.0, gs3=0.001, a=0.1)
    iters = 0
    error_table = np.zeros((5, 4), dtype=np.float)
    mesh_list = ['mesh1.npy','mesh2.npy','mesh3.npy','mesh4.npy','mesh5.npy']

    # loop over all the different mesh
    for mesh_file in mesh_list:
        fsi.print_line()
        print("\t\t\t REFINEMENT LEVEL: {}".format(iters))
        # set up fsi.OCP class
        OCP = fsi.OCP(prm, data=mesh_file, control_spec='FS_domain', tol=1e-6)
        # set up exact desired, adjoint and control variables
        (OCP.desired, ex_adjoint, ex_control, OCP.rhs) \
            = fsi.build_exact_data(OCP.mesh, OCP.tmesh, OCP.Mat, OCP.prm)
        # gradient algorithm
        (state, adjoint, control, residue) \
            = optimal_control_subroutine(OCP, NextPoint=None)
        # error in control
        res = ex_control - control
        err_control \
            = fsi.FSI_Optimality_Residual(OCP.Mat, res, OCP.mesh, OCP.prm)
        # error in state
        res = state.sliced() + OCP.desired
        err_state = res.norm(OCP.Mat, OCP.prm.tau)
        # error in adjoint
        res = ex_adjoint.invsliced() - adjoint.invsliced()
        err_adjoint = res.norm(OCP.Mat, OCP.prm.tau)
        error_table[iters, :] \
            = [OCP.mesh.size(), err_control, err_state, err_adjoint]
        display(error_table, iters)
        iters = iters + 1

    FileName = os.getcwd() + '/npyfiles/ex3_spatial_error.npy'
    np.save(FileName, {'error_table': error_table})


def myticks(x, pos):
    """
    Ticks to be used in the plot of the spatial errors.
    """
    
    s = np.log10(x)
    if s == -0.5:
        return r"$10^{-0.5}$"
    elif s == -1.0:
        return r"$10^{-1}$"
    else:
        return r"$10^{-1.5}$"


def plot_example3(_type='space'):
    """
    Plot the spatial or temporal errors in Example 3. The variable is either
    <'space'> or <'time'>.
    """

    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.unicode'] = True
    fig = plt.figure(figsize=(11,4))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.set_title(r'$\|q^* - q_{kh}\|$', fontsize=15)
    ax2.set_title(r'$\|\xi^* - \xi_{kh}\| + \|w^* - w_{kh}\|$',
        fontsize=15)
    ax3.set_title(r'$\|y^* - y_{kh}\| + \|\vartheta^* - \vartheta_{kh}\|$',
        fontsize=15)
    PathName = os.getcwd() + '/npyfiles/'

    if _type is 'space':
        FileName = PathName + 'ex3_spatial_error.npy'
    elif _type is 'time':
        FileName = PathName + 'ex3_temporal_error.npy'
    else:
        print("Option is either 'space' or 'time'")

    error_table = np.load(FileName, encoding='latin1')[()]['error_table']
    if _type is 'space':
        for it in ((1, ax1, 0.8), (2, ax2, 1.5), (3, ax3, 0.3)):
            it[1].loglog(error_table[:, 0], error_table[:, it[0]], 
                color='blue', marker='o', ms=4, lw=1)
            it[1].loglog(error_table[:, 0], (error_table[:, 0]*it[2])**2, 
                color='black', ls='-.', lw=1)
            it[1].autoscale(enable=True, axis='x', tight=True)
            it[1].autoscale(enable=True, axis='y', tight=True)
        major_ticks = [10**(-0.5), 10**(-1), 10**(-1.5)]
        for ax in (ax1, ax2, ax3):
            ax.set_xticks(major_ticks)
            ax.set_xticklabels(major_ticks)
            ax.xaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(myticks))
            plt.setp(ax.get_xticklabels(), fontsize=13)
            plt.setp(ax.get_yticklabels(), fontsize=13)
        FileName = os.getcwd() + '/figfiles/ex3_spatial_error.eps'
    elif _type is 'time':
        for it in ((1, ax1, 0.3), (2, ax2, 1.5), (3, ax3, 0.03)):
            it[1].loglog(error_table[:, 0], error_table[:, it[0]],
                color='blue', marker='o', ms=4, lw=1)
            it[1].loglog(error_table[:, 0], (error_table[:, 0]*it[2]),
                color='black', linestyle='-.', lw=1)
            it[1].autoscale(enable=True, axis='x', tight=True)
            it[1].autoscale(enable=True, axis='y', tight=True)
            it[1].tick_params(axis='both', which='major', labelsize=13)
        FileName = os.getcwd() + '/figfiles/ex3_temporal_error.eps'
    fig.savefig(FileName, format='eps', dpi=900, bbox_inches='tight')
    

def build_data_example0():
    """
    Create data for the example in the introduction.
    """

    # set up the parameters
    prm = fsi.Set_Parameters(T=2.0, r=1.0, mu=2.0, eps=0.1,
        tau=0.0025, gf=1.0, gs1=1.0, gs2=1.0, gs3=0.01, a=1e-6)

    # file name of mesh to be loaded
    mesh_file = 'mesh.npy'

    # set up the class for the optimal control problem
    OCP = fsi.OCP(prm, mesh_file, control_spec='S_domain', tol=1e-6)
    OCP.desired = fsi.build_desired_state(OCP.mesh, OCP.tmesh, OCP.init)
    OCP.tmesh.NumHistInt = 0

    # solve optimal control problem
    (state, adjoint, control, residue) \
        = optimal_control_subroutine(OCP, NextPoint='LS')

    # save data for control and state
    FileName = os.getcwd() + '/npyfiles/ex0_data.npy'
    np.save(FileName, {'state': state, 'control': control})


def plot_example0():
    """
    Plot for the example in the introduction.
    """

    # load data
    FileName = os.getcwd() + '/npyfiles/ex0_data.npy'
    state = np.load(FileName, encoding='latin1')[()]['state']
    control = np.load(FileName, encoding='latin1')[()]['control']

    # set up the parameters
    prm = fsi.Set_Parameters(T=2.0, r=1.0, mu=2.0, eps=0.1,
        tau=0.0025, gf=1.0, gs1=1.0, gs2=1.0, gs3=0.01, a=1e-6)

    # file name of mesh to be loaded
    mesh_file = 'mesh.npy'

    # set up the class for the optimal control problem
    OCP = fsi.OCP(prm, mesh_file, control_spec='S_domain', tol=1e-6)
    OCP.desired = fsi.build_desired_state(OCP.mesh, OCP.tmesh, OCP.init)

    # solve state equation with delay
    state_w_delay \
        = fsi.FSI_State_Solver(OCP.init, OCP.hist, control, OCP.Mat,
                               OCP.mesh, OCP.tmesh, OCP.prm, OCP.rhs)
    solid_vel = np.zeros((OCP.tmesh.NumNode-1,), dtype=np.float)
    solid_dsp = np.zeros((OCP.tmesh.NumNode-1,), dtype=np.float)
    res = state.sliced() - OCP.desired

    # create and save figures
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.unicode'] = True
    fig = plt.figure(figsize=(11,4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # loop over time steps
    for i in range(OCP.tmesh.NumNode-2):
        vecx = res.vel_global_x[OCP.mesh.NodeSolidIndex, i]
        vecy = res.vel_global_y[OCP.mesh.NodeSolidIndex, i]
        solid_vel[i] = np.sqrt(np.dot(vecx, OCP.Mat.Ms_block * vecx)
                               + np.dot(vecy, OCP.Mat.Ms_block * vecy))
        vecx = res.disp_solid_x[:, i]
        vecy = res.disp_solid_y[:, i]
        solid_dsp[i] = np.sqrt(np.dot(vecx, OCP.Mat.Ms_block * vecx)
                               + np.dot(vecy, OCP.Mat.Ms_block * vecy))
    time_grid = OCP.tmesh.Grid[1:len(OCP.tmesh.Grid)-1]
    ax1.plot(time_grid, solid_vel[:len(solid_vel)-1], color='blue')
    ax2.plot(time_grid, solid_dsp[:len(solid_dsp)-1], color='blue')
    ax1.set_xlim(0, 2)
    ax2.set_xlim(0, 2)
    ax1.set_title(r'$\|w_t(t) - v_d(t)\|_{\Omega_{sh}}$', fontsize=15)
    ax2.set_title(r'$\|w(t) - w_d(t)\|_{\Omega_{sh}}$', fontsize=15)

    # calculate residuals for the state with delay
    solid_vel = np.zeros((OCP.tmesh.NumNode-1,), dtype=np.float)
    solid_dsp = np.zeros((OCP.tmesh.NumNode-1,), dtype=np.float)
    resd = state_w_delay.sliced() - OCP.desired

    for i in range(OCP.tmesh.NumNode-1):
        vecx = resd.vel_global_x[OCP.mesh.NodeSolidIndex, i]
        vecy = resd.vel_global_y[OCP.mesh.NodeSolidIndex, i]
        solid_vel[i] = np.sqrt(np.dot(vecx, OCP.Mat.Ms_block * vecx)
                               + np.dot(vecy, OCP.Mat.Ms_block * vecy))
        vecx = resd.disp_solid_x[:, i]
        vecy = resd.disp_solid_y[:, i]
        solid_dsp[i] = np.sqrt(np.dot(vecx, OCP.Mat.Ms_block * vecx)
                               + np.dot(vecy, OCP.Mat.Ms_block * vecy))

    time_grid = OCP.tmesh.Grid[1:len(OCP.tmesh.Grid)-1]
    ax1.plot(time_grid, solid_vel[:len(solid_vel)-1],
             color='red', linestyle='--')
    ax2.plot(time_grid, solid_dsp[:len(solid_dsp)-1],
             color='red', linestyle='--')
    ax1.set_xlim(0, 2)
    ax2.set_xlim(0, 2)
    FileName = os.getcwd() + '/figfiles/ex0.eps'
    fig.savefig(FileName, format='eps', dpi=900, bbox_inches='tight')


def makedirs():
    """
    Create directories.
    """
    dirs = ['npyfiles', 'figfiles']
    for name in dirs:
        if not os.path.isdir(name):
            os.makedirs(name)


def print_platform():
    """
    Prints machine platform and python version.
    """
    
    string = ("PYTHON VERSION: {} \nPLATFORM: {} \nPROCESSOR: {}"
              + "\nVERSION: {} \nMAC VERSION: {}")
    print(string.format(sys.version, platform.platform(),
        platform.uname()[5], platform.version()[:60]
        + '\n' + platform.version()[60:], platform.mac_ver()))


def main():
    """
    Main function to execute all the examples. 
    """
    
    print('*'*78 + '\n')
    start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Start of Run: " + start + '\n')
    print_platform()
    print('\n')
    
    # make directories <npyfiles> and <figfiles>
    makedirs()

    # Example 0
    print('*'*78 + '\n\nEXAMPLE 0' + '\n')    
    try:
        plot_example0()
    except IOError:
        build_data_example0()
        plot_example0()     
    
    # Example 1
    print('*'*78 + '\n\nEXAMPLE 1' + '\n')    
    FileName = 'ex1_data_fluid_structure.npy'
    FigName = 'ex1_control_fluid_structure.png'
    try:
        plot_example1_control(FileName, FigName, ticksolid=False)
    except IOError:
        build_data_example1_FS()
        plot_example1_control(FileName, FigName, ticksolid=False)
    FileName = 'ex1_data_fluid.npy'
    FigName = 'ex1_control_fluid.png'
    try:
        plot_example1_control(FileName, FigName, ticksolid=False)
    except IOError:
        build_data_example1_F()
        plot_example1_control(FileName, FigName, ticksolid=False)
    FileName = 'ex1_data_structure.npy'
    FigName = 'ex1_control_structure.png'
    try:
        plot_example1_control(FileName, FigName)
    except IOError:
        build_data_example1_S()
        plot_example1_control(FileName, FigName)
    plot_example1()

    # Example 2
    print('*'*78 + '\n\nEXAMPLE 2' + '\n')
    try:
        plot_example2()
    except IOError:
        build_data_example2()
        plot_example2()

    # Example 3
    print('*'*78 + '\n\nEXAMPLE 3' + '\n')
    for _type in ['time', 'space']:
        try:
            plot_example3(_type)
        except IOError:
            if _type is 'space':
                build_data_example3_space()
            elif _type is 'time':
                build_data_example3_time()
            plot_example3(_type)

    print('\n')
    end = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print('\n' + "End of Run: " + end)
    print('*'*78 + '\n\n')

if __name__ == '__main__':
    main()
