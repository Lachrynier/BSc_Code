### This file containts functions that were used to generate the majority of plots for the report

import sys
import os

# Change paths to accomodate your setup
sys.path.append('/dtu/3d-imaging-center/projects/2022_DANFIX_Vindelev/analysis/s214743_bsc')
base_dir = os.path.abspath('/dtu/3d-imaging-center/projects/2022_DANFIX_Vindelev/analysis/s214743_bsc/')
bfig_dir = os.path.join(base_dir,'bjobs/figs')


### General imports
import math
import spekpy
import pickle
from time import time

import numpy as np
from numpy.linalg import solve

import scipy as sp
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.ndimage import convolve1d

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,LogNorm

import skimage
from skimage import color, io, measure, img_as_ubyte, img_as_float
from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import disk
from skimage.measure import profile_line
from skimage.filters import threshold_otsu

### CIL imports
import cil
from cil.framework import ImageData, ImageGeometry, AcquisitionData, AcquisitionGeometry, VectorData, VectorGeometry
from cil.framework import DataContainer, BlockDataContainer
from cil.io import NikonDataReader
from cil.processors import TransmissionAbsorptionConverter, Slicer, AbsorptionTransmissionConverter
from cil.utilities.jupyter import islicer
from cil.utilities.display import show_geometry, show2D, show1D
from cil.recon import FDK, FBP
from cil.optimisation.algorithms import GD, FISTA, PDHG, SIRT
from cil.optimisation.operators import CompositionOperator, FiniteDifferenceOperator, MatrixOperator, LinearOperator, \
                                       BlockOperator, GradientOperator
from cil.optimisation.functions import IndicatorBox, MixedL21Norm, L2NormSquared, \
                                       BlockFunction, L1Norm, LeastSquares, \
                                       OperatorCompositionFunction, TotalVariation, \
                                       ZeroFunction
from cil.plugins.tigre import ProjectionOperator
from cil.plugins.ccpi_regularisation.functions import FGP_TV


### My imports
from core import fun_attenuation, generate_spectrum
from core import generate_triangle_image, create_circle_image, create_rings_image
from core import load_centre, BHC
from core import lin_interp_sino2D, make_projection_plot
from core import compute_scatter_basis, ESCMatrixOperator


def spectrum_penetration_plot():
    mu = fun_attenuation(plot=False)
    bin_centers, bin_heights = generate_spectrum(plot=False,filter=0,tube_potential=200,bin_width=0.01)
    num_bins = bin_centers.size

    max_length = 10
    # d = np.linspace(0,max_length, dtype='float64')
    d = np.array([0,0.4,2.5], dtype='float64')
    I_E = np.zeros((num_bins,d.size), dtype='float64')
    for i in range(num_bins):
        E = bin_centers[i]
        I0_E = bin_heights[i]
        I_E[i,:] = I0_E * np.exp(-mu(E)*d)
    
    I_E = I_E / np.sum(I_E, axis=0)

    bin_edges = np.linspace(0, 200, 81)  # Creates 80 bins from 0 to 200

    plt.figure(figsize=(10, 6))  # Set the figure size
    # kwargs = dict(alpha=0.5, bins=80, density=True, stacked=True)  # Common kwargs for all histograms
    kwargs = dict(alpha=0.5, density=True, stacked=True)  # Removed 'bins=80'
    # plt.hist(bin_centers, **kwargs, weights=I_E[:,0], color='r', label=fr'$d = 0$ mm')
    # plt.hist(bin_centers, **kwargs, weights=I_E[:,1], color='g', label=fr'$d = {d[1]}$ mm')
    # plt.hist(bin_centers, **kwargs, weights=I_E[:,2], color='b', label=fr'$d = {d[2]}$ mm')

    plt.hist(bin_centers, bins=bin_edges, **kwargs, weights=I_E[:,0], color='r', edgecolor='r', label=fr'$d = 0$ mm')
    plt.hist(bin_centers, bins=bin_edges, **kwargs, weights=I_E[:,1], color='g', edgecolor='g', label=fr'$d = {d[1]}$ mm')
    plt.hist(bin_centers, bins=bin_edges, **kwargs, weights=I_E[:,2], color='b', edgecolor='b', label=fr'$d = {d[2]}$ mm')

    plt.xlabel(r'$E$ [keV]')
    plt.title(r'Normalized X-ray spectrum histograms at different path lengths $d$')
    plt.legend()
    plt.tight_layout()
    # plt.savefig(os.path.join(base_dir, 'plots/spectrum_snapshots.pdf'))
    plt.show()
    print(np.sum(I_E[:,0]))
    print(np.sum(I_E[:,2]))

def random_plot():
    # For testing plt tex capabilities
    from matplotlib import rc
    import matplotlib
    d = np.linspace(0, 10, 100)  # Replace with your actual data
    mean_energies = np.sin(d)  # Replace with your actual data

    # Set up Matplotlib to use LaTeX for text rendering
    # plt.rcParams.update({
    #     "text.usetex": True,
    #     "font.family": "serif",
    #     "font.serif": ["Palatino"],
    # })

    # font = {'family' : 'monospace',
    #     'weight' : 'bold',
    #     'size'   : 22}
    # rc('font', **font)

    plt.rcParams.update({"text.usetex": True})
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    matplotlib.rcParams['mathtext.fontset'] = 'stix'

    # Create the plot
    plt.figure(figsize=(8, 6))  # Choose a good size for thesis or papers
    plt.plot(d, mean_energies, label=r'Mean Energies')  # Use raw string for LaTeX
    plt.title(r'\textbf{Energy Distribution}', fontsize=16)  # Bold title with LaTeX
    plt.xlabel(r'Distance ($m$)', fontsize=14)  # Label with units
    plt.ylabel(r'Energy $(J)$ $\sum \xi$', fontsize=14)  # Label with units
    plt.legend(frameon=False, fontsize=12)
    plt.grid(True)
    plt.tight_layout()  # Adjust the layout to not cut off anything
    
    # Set font size of the numbers on the x and y axes
    plt.tick_params(axis='both', which='major', labelsize=12)
    # plt.tick_params(axis='both', which='minor', labelsize=10)  # If using minor ticks
    # plt.savefig('energy_distribution.pdf')
    plt.show()

def mean_energy_plot():
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
        "font.serif": ["Computer Modern"],  # This is the standard LaTeX font
        "font.size": 14,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })
    mu = fun_attenuation(plot=False)
    bin_centers, bin_heights = generate_spectrum(plot=False,filter=0,tube_potential=200,bin_width=0.5)
    num_bins = bin_centers.size


    fig, ax = plt.subplots(1, 3, figsize=(12, 5))

    max_lengths = [0.1,2.5,15]
    for k,max_length in enumerate(max_lengths):
        d = np.linspace(0,max_length,num=300, dtype='float64')
        I_E = np.zeros((num_bins,d.size), dtype='float64')
        for i in range(num_bins):
            E = bin_centers[i]
            I0_E = bin_heights[i]
            I_E[i,:] = I0_E * np.exp(-mu(E)*d)
        
        I_E = I_E / np.sum(I_E, axis=0)
        
        mean_energies = np.sum(bin_centers[:,None] * I_E, axis=0)

        ax[k].set_ylabel(r'[keV]')
        ax[k].set_xlabel(r'$d$ [mm]')
        # ax[k].set_title(r'$d\in [0,10]$')
        ax[k].set_title(fr'$d\in [0,{max_length}]$')
        ax[k].plot(d,mean_energies)
        ax[k].grid(True)
    
    ax[2].set_xticks(np.linspace(start=0,stop=15,num=7))
    # plt.xscale('log')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    fig.suptitle(r'Mean energy of the X-ray spectrum as a function of the path length $d$')
    plt.savefig(os.path.join(base_dir, 'plots/mean_energy.pdf'))
    plt.show()

def attenuation_plots():
    # mu/rho in (cm2/g)
    # multiply by density of gold to get mu in (cm^{-1})
    # ORIGINAL DATA IN MEV!!!! SO CONVERT TO KEV
    data = []
    file_path = os.path.join(base_dir,'NIST_gold_only_dat.txt')
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into components
            parts = line.split()

            # Check if the line has at least 3 elements and the first is a number
            if len(parts) >= 3 and parts[0].replace('.', '', 1).replace('E-', '', 1).replace('E+', '', 1).isdigit():
                energy = float(parts[-3])  # Energy value
                mu_rho = float(parts[-2])  # mu/rho value
                mu_en_rho = float(parts[-1])  # mu_en/rho value
                data.append((energy, mu_rho, mu_en_rho))

    data = np.array(data)

    # convert energies to keV
    energies = 1000*data[:,0]

    # calculate mu = mu/rho * rho
    rho_gold = 19.3 # g/cm^3
    # rho_gold = 1
    # rho_gold = 1
    print(f'rho: {rho_gold}')
    # divide by 10 to get it in mm
    mu = data[:,1] * rho_gold / 10 # mm^{-1}

    # Perform spline interpolation in log domain
    spline_log_domain = interpolate.InterpolatedUnivariateSpline(np.log10(energies), np.log10(mu), k=1)
    def estimate_attenuation(energy):
        return np.power(10, spline_log_domain(np.log10(energy)))
    
    energy_plot_range = np.logspace(np.log10(min(energies)), np.log10(max(energies)), 500)
    ##############################
    #####
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
        "font.serif": ["Computer Modern"],  # This is the standard LaTeX font
        "font.size": 14,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    ###
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].loglog(energy_plot_range, estimate_attenuation(energy_plot_range), label='Interpolation')
    ax[0].loglog(energies, mu, '.', label='Raw data',markersize=5,color='red')  # Data points

    ax[0].set_xlabel(r'$E$ [keV]')
    ax[0].set_ylabel(r'$\mu$ [mm$^{-1}$]')
    ax[0].set_title('Complete range of raw data')
    ax[0].grid(True)
    ax[0].legend()
    
    mu = fun_attenuation(plot=False)
    bin_centers, bin_heights = generate_spectrum(plot=False)
    num_bins = bin_centers.size

    ax[1].semilogy(bin_centers, mu(bin_centers))
    ax[1].set_title('Range narrowed down to energies in gold scans')
    
    ax[1].set_xlabel(r'$E$ [keV]')
    ax[1].set_ylabel(r'$\mu$ [mm$^{-1}$]')
    ax[1].grid(True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    fig.suptitle(r'Gold attenuation coefficient $\mu$ as a function of energy $E$')
    plt.savefig(os.path.join(base_dir, 'plots/gold_attenuations.pdf'))
    plt.show()

def transmission_plot():
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
        "font.serif": ["Computer Modern"],  # This is the standard LaTeX font
        "font.size": 14,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })
    energies = [150,220]
    mu = fun_attenuation(plot=False)
    max_pen = 10
    xx = np.linspace(0,max_pen,num=200)
    plt.figure(figsize=(8,8))
    for E in energies:
        mu_E = mu(E)
        plt.plot(xx,np.exp(-mu_E*xx), label=f'{E} keV')
    plt.legend()
    plt.xlabel(r'Path length [mm]')
    plt.title(r'Transmission $I/I_0$ at fixed energies')
    plt.yscale('log')
    plt.grid(True)

    plt.yticks(10.0**np.arange(-15, 1, 1))  # Corrected line here
    plt.xticks(np.arange(0,max_pen+1,1))
    # plt.grid(True, which="both", ls="--")  # This ensures grid lines are drawn for both major and minor ticks
    # plt.savefig(os.path.join(base_dir, 'plots/trans_bounds.pdf'))
    plt.show()

def staircase_plot():
    physical_in_mm = 1
    voxel_num = 20
    angles = np.linspace(start=0, stop=180, num=1, endpoint=False)
    physical_size = physical_in_mm # mm
    voxel_size = physical_size/voxel_num

    ig = ImageGeometry(voxel_num_x=voxel_num, voxel_num_y=voxel_num, voxel_size_x=voxel_size, voxel_size_y=voxel_size, center_x=0, center_y=0)

    factor = 1
    panel_num_cells = math.ceil(factor*voxel_num)
    panel_cell_length = 1/factor * voxel_size
    ag = AcquisitionGeometry.create_Parallel2D(ray_direction=[0,1], detector_position=[0,physical_size], detector_direction_x=[1,0], rotation_axis_position=[0,0])\
        .set_panel(num_pixels=panel_num_cells,pixel_size=panel_cell_length)\
        .set_angles(angles=angles)

    tri_width = voxel_num
    corner = 0

    img_size = (voxel_num, voxel_num)  # Image size: 100x100 pixels
    triangle_size = (tri_width, tri_width)  # Triangle size: base=50, height=50 pixels
    corner_coords = (corner, corner)  # Right-angled corner at (25, 25)
    triangle_image = generate_triangle_image(img_size, triangle_size, corner_coords)
    im_arr = np.rot90(triangle_image, 1)
    im_arr = im_arr.astype('float32')
    # plt.imshow(im_arr,cmap='gray')
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()

    image = ImageData(array=im_arr.T,geometry=ig)
    show2D(image)
    show2D(image, title=r'Staircase test image', size=(10,10)).save(os.path.join(base_dir, 'plots/staircase_test_image.pdf'))

def staircase_experiments():
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
        "font.serif": ["Computer Modern"],  # This is the standard LaTeX font
        "font.size": 14,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    def staircase_bhc(physical_in_mm,voxel_num=1000,mono_E=None):
        ### Set up CIL geometries
        angles = np.linspace(start=0, stop=180, num=1, endpoint=False)
        physical_size = physical_in_mm # mm
        voxel_size = physical_size/voxel_num

        ig = ImageGeometry(voxel_num_x=voxel_num, voxel_num_y=voxel_num, voxel_size_x=voxel_size, voxel_size_y=voxel_size, center_x=0, center_y=0)

        factor = 1
        panel_num_cells = math.ceil(factor*voxel_num)
        panel_cell_length = 1/factor * voxel_size
        ag = AcquisitionGeometry.create_Parallel2D(ray_direction=[0,1], detector_position=[0,physical_size], detector_direction_x=[1,0], rotation_axis_position=[0,0])\
            .set_panel(num_pixels=panel_num_cells,pixel_size=panel_cell_length)\
            .set_angles(angles=angles)

        plot_size = 6
        # show_geometry(ag, ig, grid=True, figsize=(plot_size, plot_size),fontsize=plot_size)

        ### Generate staircase test image
        tri_width = voxel_num
        corner = 0

        img_size = (voxel_num, voxel_num)  # Image size: 100x100 pixels
        triangle_size = (tri_width, tri_width)  # Triangle size: base=50, height=50 pixels
        corner_coords = (corner, corner)  # Right-angled corner at (25, 25)
        triangle_image = generate_triangle_image(img_size, triangle_size, corner_coords)
        im_arr = np.rot90(triangle_image, 1)
        im_arr = im_arr.astype('float32')
        # plt.imshow(im_arr,cmap='gray')
        # plt.xticks([])
        # plt.yticks([])
        # plt.show()

        image = ImageData(array=im_arr.T,geometry=ig)
        # show2D(image, size=(8,8))

        ###
        mu = fun_attenuation(plot=False)
        bin_centers, bin_heights = generate_spectrum(plot=False)
        num_bins = bin_centers.size

        A = ProjectionOperator(ig, ag, 'Siddon', device='gpu')
        d = A.direct(image)
        d = d.as_array()
        I = np.zeros(d.shape, dtype='float64')
        I0 = 0
        for i in range(num_bins):
            E = bin_centers[i]
            I0_E = bin_heights[i]
            I0 += I0_E
            I += I0_E * np.exp(-mu(E)*d)

        print(f'minmax D: {d.min()}, {d.max()}')
        print(f'minmax I: {I.min()}, {I.max()}')

        I = np.array(I,dtype='float32')
        data = AcquisitionData(array=-np.log(I/I0), geometry=ag)
        b = data.as_array()

        if mono_E is None:
            mu_eff = np.sum(bin_heights * mu(bin_centers))
            b_mono = d*mu_eff
        else:
            b_mono = d*mu(mono_E)
        

        # x = voxel_size * np.arange(1,b.size+1,1)
        # y = b_mono-b
        # corrections = np.column_stack((x,y))
        # plt.plot(b,b_mono-b)
        # plt.title('b_mono-b vs b')
        # plt.show()

        # plt.plot(voxel_size * np.arange(1,b.size+1,1),b_mono-b)
        # plt.title('b_mono-b vs path length')
        # plt.show()

        # corrections2 = np.column_stack((b,b_mono-b))
        return b,b_mono

    max_lengths = [0.01,0.1] # mm
    voxel_num = 1000
    fig, ax = plt.subplots(1, len(max_lengths), figsize=(10, 5))
    for k,max_length in enumerate(max_lengths):
        voxel_size = max_length/voxel_num
        b,b_mono = staircase_bhc(physical_in_mm=max_length,voxel_num=voxel_num,mono_E=None)
        ax[k].set_xlabel(r'$d$ [mm]')
        ax[k].set_title(fr'$d\in [0,{max_length}]$')
        ax[k].plot(voxel_size * np.arange(1,b_mono.size+1,1),b_mono,'--',label=r'Monochromatic absorption at $\mu_{\text{eff}}(0)$')
        ax[k].plot(voxel_size * np.arange(1,b.size+1,1), b, label=r'Polychromatic absorption')
        ax[k].grid(True)
        ax[k].legend()
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    fig.suptitle(r'Beam hardening effect on absorption as a function of the path length $d$')
    plt.savefig(os.path.join(base_dir, 'plots/bh_effect.pdf'))
    plt.show()

# ax[k].set_ylabel(r'[keV]')
#         ax[k].set_xlabel(r'$d$ [mm]')
#         # ax[k].set_title(r'$d\in [0,10]$')
#         ax[k].set_title(fr'$d\in [0,{max_length}]$')
#         ax[k].plot(d,mean_energies)
#         ax[k].grid(True)
    
#     ax[2].set_xticks(np.linspace(start=0,stop=15,num=7))
#     # plt.xscale('log')
#     plt.tight_layout()
#     plt.subplots_adjust(top=0.85)
#     fig.suptitle(r'Mean energy of the X-ray spectrum as a function of the path length $d$')
#     plt.savefig(os.path.join(base_dir, 'plots/mean_energy.pdf'))
#     plt.show()




########## bh
def setup_generic_cil_geometry(physical_size, voxel_num, cell_to_im_ratio=1, fan=False):
### Set up CIL geometries
    angles = np.linspace(start=0, stop=180, num=3*180//1, endpoint=False)
    voxel_size = physical_size/voxel_num
    ig = ImageGeometry(voxel_num_x=voxel_num, voxel_num_y=voxel_num, voxel_size_x=voxel_size, voxel_size_y=voxel_size, center_x=0, center_y=0)
    
    factor = cell_to_im_ratio
    panel_num_cells = math.ceil(np.sqrt(2)*factor*voxel_num)
    panel_cell_length = 1/factor * voxel_size

    # if fan:
    #     angles = np.linspace(start=0, stop=360, num=10*180//1, endpoint=False)
    #     diag_size = math.ceil(np.sqrt(2))
    #     dist_factor = 5
    #     ag = AcquisitionGeometry.create_Cone2D(source_position=[0,-2*physical_size], detector_position=[0,dist_factor*physical_size], detector_direction_x=[1,0], rotation_axis_position=[0,0])\
    #         .set_panel(num_pixels=panel_num_cells,pixel_size=5*panel_cell_length,origin='top-right')\
    #         .set_angles(angles=angles)
    #     # ag = AcquisitionGeometry.create_Cone2D(source_position=[0,-2*physical_size], detector_position=[0,2*physical_size])\
    #     #     .set_panel(num_pixels=2*diag_size*panel_num_cells, pixel_size=panel_cell_length)\
    #     #     .set_angles(angles=angles)
    # else:
    ag = AcquisitionGeometry.create_Parallel2D(ray_direction=[0,1], detector_position=[0,physical_size], detector_direction_x=[1,0], rotation_axis_position=[0,0])\
        .set_panel(num_pixels=panel_num_cells,pixel_size=panel_cell_length)\
        .set_angles(angles=angles)
    
    return ag,ig
def generate_bh_data(im, ag, ig):
    mu = fun_attenuation(plot=False)
    bin_centers, bin_heights = generate_spectrum(plot=False)
    num_bins = bin_centers.size
    A = ProjectionOperator(ig, ag, 'Siddon', device='gpu')
    d = A.direct(im)
    d = d.as_array()
    I = np.zeros(d.shape, dtype='float32')
    I0 = 0
    # print(d)
    for i in range(num_bins):
        E = bin_centers[i]
        I0_E = bin_heights[i]
        I0 += I0_E
        I += I0_E * np.exp(-mu(E)*d)
    
    b = AcquisitionData(array=-np.log(I/I0), geometry=ag)
    return b

def bh_disk():
    physical_size = 1
    voxel_num = 1000
    ag,ig = setup_generic_cil_geometry(physical_size=1,voxel_num=voxel_num)
    im_arr = create_circle_image(image_size=voxel_num, radius=voxel_num//2.5, center=[voxel_num//2, voxel_num//2])
    im = ImageData(array=im_arr.astype('float32'), geometry=ig)
    b = generate_bh_data(im, ag, ig)
    recon = FBP(b, image_geometry=ig).run(verbose=0)

############# Plotting
    from matplotlib.gridspec import GridSpec,GridSpecFromSubplotSpec
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 1, height_ratios=[3, 1])

    # Create a nested 1x2 grid in the first part of the 2x1 grid
    top_row_gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0])
    ax = [None] * 3
    ax[0] = fig.add_subplot(top_row_gs[0, 0])
    ax[1] = fig.add_subplot(top_row_gs[0, 1])
    ax[2] = fig.add_subplot(gs[1])

    plt.sca(ax[0])
    plt.imshow(im_arr, origin='lower',cmap='grey')
    plt.xlabel('horizontal_x')
    plt.ylabel('horizontal_y')
    plt.title('Disk test image')
    plt.colorbar()

    plt.sca(ax[1])
    plt.imshow(recon.as_array(), origin='lower', cmap='gray')
    plt.xlabel('horizontal_x')
    plt.ylabel('horizontal_y')
    plt.title('FBP reconstruction')
    plt.colorbar()

    plt.sca(ax[2])
    plt.plot(recon.as_array()[recon.shape[0]//2,:])
    plt.xlabel('horizontal_x')
    plt.title(f'Intensity profile for horizontal_y={recon.shape[0]//2} on the reconstruction')
    plt.grid(True)
    plt.tight_layout()
    # plt.subplots_adjust(top=0.85)
    # plt.savefig(os.path.join(base_dir, 'plots/bh_disk.pdf'))
    plt.show()
################

def bh_rings():
    physical_size = 1
    voxel_num = 1000
    ag,ig = setup_generic_cil_geometry(physical_size=1,voxel_num=voxel_num)


    im_arr = create_rings_image(size=voxel_num, spacing=40, ring_width=100, radius=voxel_num//3, center=[voxel_num//2, voxel_num//2])

    # im1 = create_circle_image(image_size=voxel_num, radius=voxel_num//2.5, center=[voxel_num//2, voxel_num//2])
    # im2 = create_circle_image(image_size=voxel_num, radius=voxel_num//3.5, center=[voxel_num//2, voxel_num//2])
    # im_arr = im1-im2

    im = ImageData(array=im_arr.astype('float32'), geometry=ig)
    b = generate_bh_data(im, ag, ig)
    recon = FBP(b, image_geometry=ig).run(verbose=0)

############# Plotting
    from matplotlib.gridspec import GridSpec,GridSpecFromSubplotSpec
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 1, height_ratios=[3, 1])

    # Create a nested 1x2 grid in the first part of the 2x1 grid
    top_row_gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0])
    ax = [None] * 3
    ax[0] = fig.add_subplot(top_row_gs[0, 0])
    ax[1] = fig.add_subplot(top_row_gs[0, 1])
    ax[2] = fig.add_subplot(gs[1])

    plt.sca(ax[0])
    plt.imshow(im_arr, origin='lower',cmap='grey')
    plt.xlabel('horizontal_x')
    plt.ylabel('horizontal_y')
    plt.title('Concentric rings test image')
    plt.colorbar()

    plt.sca(ax[1])
    plt.imshow(recon.as_array(), origin='lower', cmap='gray')
    plt.xlabel('horizontal_x')
    plt.ylabel('horizontal_y')
    plt.title('FBP reconstruction')
    plt.colorbar()

    plt.sca(ax[2])
    plt.plot(recon.as_array()[recon.shape[0]//2,:])
    plt.xlabel('horizontal_x')
    plt.title(f'Intensity profile for horizontal_y={recon.shape[0]//2} on the reconstruction')
    plt.grid(True)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(os.path.join(base_dir, 'plots/bh_rings.pdf'))
    plt.show()


def bh_slit():
    physical_size = 1
    voxel_num = 1000
    ag,ig = setup_generic_cil_geometry(physical_size=1,voxel_num=voxel_num)

    im_arr = io.imread(os.path.join(base_dir,'test_images/test_slit_3.png'))
    im_arr = color.rgb2gray(im_arr) > 0

    im = ImageData(array=im_arr.astype('float32'), geometry=ig)
    b = generate_bh_data(im, ag, ig)
    recon = FBP(b, image_geometry=ig).run(verbose=0)

############# Plotting
    from matplotlib.gridspec import GridSpec,GridSpecFromSubplotSpec
    fig = plt.figure(figsize=(10,14))
    gs = GridSpec(4, 1, height_ratios=[1.5,1.5, 1, 1])

    # Create a nested 1x2 grid in the first part of the 2x1 grid
    # top_row_gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0])
    ax = [None] * 4
    ax[0] = fig.add_subplot(gs[0])
    ax[1] = fig.add_subplot(gs[1])
    ax[2] = fig.add_subplot(gs[2])
    ax[3] = fig.add_subplot(gs[3])

    hy_range = [350,600]
    # hy_range = [0,1000]
    hx_range = [100,900]
    plt.sca(ax[0])
    im0 = plt.imshow(im_arr[hy_range[0]:hy_range[1],hx_range[0]:hx_range[1]], origin='lower',cmap='grey')
    # plt.imshow(im_arr, origin='lower',cmap='grey')
    plt.xlabel('horizontal_x')
    plt.ylabel('horizontal_y')
    plt.title('Slit test image')
    # plt.colorbar()
    plt.colorbar(im0,fraction=0.046, pad=0.04)

    plt.sca(ax[1])
    im1 = plt.imshow(recon.as_array()[hy_range[0]:hy_range[1],hx_range[0]:hx_range[1]], origin='lower', cmap='gray')#,vmin=-10, vmax=10)
    # plt.imshow(recon.as_array(), origin='lower', cmap='gray')
    plt.xlabel('horizontal_x')
    plt.ylabel('horizontal_y')
    plt.title('FBP reconstruction')
    # plt.colorbar()
    plt.colorbar(im1,fraction=0.046, pad=0.04)

    hori_idx = 125
    hori_idx += hy_range[0]
    plt.sca(ax[2])
    # plt.plot(recon.as_array()[:,hori_idx]) # horizontal_x fixed
    plt.plot(recon.as_array()[hori_idx,hx_range[0]:hx_range[1]]) # horizontal_y fixed
    plt.xlabel('horizontal_x')
    plt.title(f'Intensity profile for horizontal_y={hori_idx-hy_range[0]} on the reconstruction')
    plt.grid(True)

    # hori_idx = 75
    # hori_idx += hy_range[0]
    # plt.sca(ax[3])
    # # plt.plot(recon.as_array()[:,hori_idx]) # horizontal_x fixed
    # plt.plot(recon.as_array()[hori_idx,hx_range[0]:hx_range[1]]) # horizontal_y fixed
    # plt.xlabel('horizontal_x')
    # plt.title(f'Intensity profile for horizontal_y={hori_idx-hy_range[0]} on the reconstruction')
    # plt.grid(True)

    # hori_idx = 390
    # hori_idx = 140
    hori_idx = 645
    hori_idx += hx_range[0]
    plt.sca(ax[3])
    # plt.plot(recon.as_array()[:,hori_idx]) # horizontal_x fixed
    plt.plot(recon.as_array()[hy_range[0]:hy_range[1],hori_idx]) # horizontal_y fixed
    plt.xlabel('horizontal_y')
    plt.title(f'Intensity profile for horizontal_y={hori_idx-hx_range[0]} on the reconstruction')
    plt.grid(True)

    # plt.sca(ax[3])
    # # plt.plot(im.as_array()[:,hori_idx]) # horizontal_x fixed
    # plt.plot(im.as_array()[hori_idx,:]) # horizontal_y fixed
    # plt.xlabel('horizontal_x')
    # plt.title(f'Intensity profile for horizontal_y={hori_idx-hy_range[0]} on the original image')
    # plt.grid(True)

    plt.tight_layout(pad=0, h_pad=1.3)
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.85)
    plt.savefig(os.path.join(base_dir, 'plots/bh_slit.pdf'))
    plt.show()

def bh_shapes():
    physical_size = 1
    voxel_num = 1000
    ag,ig = setup_generic_cil_geometry(physical_size=1,voxel_num=voxel_num)

    im_arr = io.imread(os.path.join(base_dir,'test_images/test_image_shapes3.png'))
    im_arr = color.rgb2gray(im_arr) > 0

    im = ImageData(array=im_arr.astype('float32'), geometry=ig)
    b = generate_bh_data(im, ag, ig)
    recon = FBP(b, image_geometry=ig).run(verbose=0)

############# Plotting
    from matplotlib.gridspec import GridSpec,GridSpecFromSubplotSpec
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(3, 1, height_ratios=[3, 1, 1])

    # Create a nested 1x2 grid in the first part of the 2x1 grid
    top_row_gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0])
    ax = [None] * 4
    ax[0] = fig.add_subplot(top_row_gs[0, 0])
    ax[1] = fig.add_subplot(top_row_gs[0, 1])
    ax[2] = fig.add_subplot(gs[1])
    ax[3] = fig.add_subplot(gs[2])


    plt.sca(ax[0])
    plt.imshow(im_arr, origin='lower',cmap='grey')
    plt.xlabel('horizontal_x')
    plt.ylabel('horizontal_y')
    plt.title('Shapes test image')
    plt.colorbar()

    plt.sca(ax[1])
    plt.imshow(recon.as_array(), origin='lower', cmap='gray')
    plt.xlabel('horizontal_x')
    plt.ylabel('horizontal_y')
    plt.title('FBP reconstruction')
    plt.colorbar()

    hori_idx = 650
    plt.sca(ax[2])
    # plt.plot(recon.as_array()[:,hori_idx]) # horizontal_x fixed
    plt.plot(recon.as_array()[hori_idx,:]) # horizontal_y fixed
    plt.xlabel('horizontal_x')
    plt.title(f'Intensity profile for horizontal_y={hori_idx} on the reconstruction')
    plt.grid(True)

    plt.sca(ax[3])
    plt.plot(im.as_array()[hori_idx,:]) # horizontal_y fixed
    plt.xlabel('horizontal_x')
    plt.title(f'Intensity profile for horizontal_y={hori_idx} on the original image')
    plt.grid(True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    # plt.savefig(os.path.join(base_dir, 'plots/bh_shapes.pdf'))
    plt.show()

def bh_shapes_v2():
    physical_size = 1
    voxel_num = 1000
    ag,ig = setup_generic_cil_geometry(physical_size=1,voxel_num=voxel_num)

    im_arr = io.imread(os.path.join(base_dir,'test_images/test_image_shapes3.png'))
    im_arr = color.rgb2gray(im_arr) > 0

    im = ImageData(array=im_arr.astype('float32'), geometry=ig)
    b = generate_bh_data(im, ag, ig)
    recon = FBP(b, image_geometry=ig).run(verbose=0)

############# Plotting
    from matplotlib import rc
    plt.rcParams.update({'font.size': 16})
    rc('text', usetex=True)
    rc('font', family='serif')


    from matplotlib.gridspec import GridSpec,GridSpecFromSubplotSpec
    fig = plt.figure(figsize=(10,18))
    gs = GridSpec(4, 1, height_ratios=[4, 4, 1, 1])

    ax = [None] * 4
    ax[0] = fig.add_subplot(gs[0])
    ax[1] = fig.add_subplot(gs[1])
    ax[2] = fig.add_subplot(gs[2])
    ax[3] = fig.add_subplot(gs[3])

    plt.sca(ax[0])
    plt.imshow(im_arr, origin='lower',cmap='grey')#, aspect='auto')
    plt.xlabel('horizontal_x')
    plt.ylabel('horizontal_y')
    plt.title('Shapes test image')
    # ax[0].set_aspect('equal', adjustable='box')
    # ax[0].set_anchor('C')
    plt.colorbar()

    plt.sca(ax[1])
    plt.imshow(recon.as_array(), origin='lower', cmap='gray')
    plt.xlabel('horizontal_x')
    plt.ylabel('horizontal_y')
    plt.title('FBP reconstruction')
    plt.colorbar()

    hori_idx = 650
    plt.sca(ax[2])
    # plt.plot(recon.as_array()[:,hori_idx]) # horizontal_x fixed
    plt.plot(recon.as_array()[hori_idx,:]) # horizontal_y fixed
    plt.xlabel('horizontal_x')
    plt.title(f'Intensity profile for horizontal_y={hori_idx} on the reconstruction')
    plt.grid(True)

    plt.sca(ax[3])
    plt.plot(im.as_array()[hori_idx,:]) # horizontal_y fixed
    plt.xlabel('horizontal_x')
    plt.title(f'Intensity profile for horizontal_y={hori_idx} on the original image')
    plt.grid(True)

    plt.tight_layout(pad=0, h_pad=1.3)
    # plt.subplots_adjust(top=0.85)
    # plt.subplots_adjust(top=0.95, bottom=0.05)
    plt.savefig(os.path.join(base_dir, 'plots/bh_shapes.pdf'))
    plt.show()

def theoretical_bhc(physical_size,mono_E=None,num_samples=1000):
    ###
    mu = fun_attenuation(plot=False)
    bin_centers, bin_heights = generate_spectrum(plot=False)
    num_bins = bin_centers.size

    O = {}
    d_step_size = physical_size/num_samples
    d = np.linspace(0, physical_size, num_samples)
    I = np.zeros(d.shape, dtype='float64')
    I0 = 0
    for i in range(num_bins):
        E = bin_centers[i]
        I0_E = bin_heights[i]
        I0 += I0_E
        I += I0_E * np.exp(-mu(E)*d)

    I = np.array(I,dtype='float32')
    b = -np.log(I/I0)
    if mono_E is None:
        mu_eff = np.sum(bin_heights * mu(bin_centers))
        print(f"mu_eff: {mu_eff}")
        b_mono = d*mu_eff
    else:
        b_mono = d*mu(mono_E)
    # x = d_step_size * np.arange(1,b.size+1,1)
    # y = b_mono-b
    # corrections = np.column_stack((x,y))
    # plt.plot(b,b_mono-b)
    # plt.title('b_mono-b vs b')
    # plt.show()
        
    
    spline_corrections = interpolate.InterpolatedUnivariateSpline(b, b_mono, k=1)
    # return corrections,corrections2

    O['d_step_size'] = d_step_size
    O['b'] = b
    O['b_mono'] = b_mono
    return spline_corrections,O
def bhc_shapes():
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
        "font.serif": ["Computer Modern"],  # This is the standard LaTeX font
        "font.size": 14,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    physical_size = 1
    voxel_num = 1000
    ag,ig = setup_generic_cil_geometry(physical_size=physical_size,voxel_num=voxel_num)

    im_arr = io.imread(os.path.join(base_dir,'test_images/test_image_shapes3.png'))
    im_arr = color.rgb2gray(im_arr) > 0

    im = ImageData(array=im_arr.astype('float32'), geometry=ig)
    b = generate_bh_data(im, ag, ig)
    recon = FBP(b, image_geometry=ig).run(verbose=0)

    tau = threshold_otsu(recon.as_array())
    segmented = ImageData(array=np.array(recon > tau, dtype='float32'), geometry=ig)
    # show2D(segmented)

    bhc,O = theoretical_bhc(physical_size,mono_E=None,num_samples=1000)
    max_idx = O['b'].size
    # plt.plot(O['b'][:max_idx],O['b_mono'][:max_idx])
    # plt.xlabel('$b$')
    # plt.ylabel('$\overline{b}$')
    # plt.show()

    b_bar = AcquisitionData(array=np.array(bhc(b.as_array()),dtype='float32'), geometry=ag)
    recon_bhc = FBP(b_bar, image_geometry=ig).run(verbose=0)
    # show2D(recon_bhc, title='recon_bhc')
    # show1D(recon_bhc, slice_list=[('horizontal_y', recon.shape[0]//2)])

    # A = ProjectionOperator(ig, ag, 'Siddon', device='gpu')
    # path_lengths = A.direct(segmented)
    # plt.plot(path_lengths,b_bar)

#######
    from matplotlib.gridspec import GridSpec,GridSpecFromSubplotSpec
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 1, height_ratios=[3, 1])

    # Create a nested 1x2 grid in the first part of the 2x1 grid
    top_row_gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0])
    ax = [None] * 3
    ax[0] = fig.add_subplot(top_row_gs[0, 0])
    ax[1] = fig.add_subplot(top_row_gs[0, 1])
    ax[2] = fig.add_subplot(gs[1])

    plt.sca(ax[0])
    plt.plot(O['b'][:max_idx],O['b_mono'][:max_idx])
    plt.grid(True)
    plt.xlabel('$b$')
    plt.ylabel('$\overline{b}$')
    plt.title(r'Linear interpolation of $b\mapsto\overline{b}$')

    plt.sca(ax[1])
    plt.imshow(recon_bhc.as_array(), origin='lower', cmap='gray')
    plt.xlabel('horizontal_x')
    plt.ylabel('horizontal_y')
    plt.title('FBP reconstruction of BHC data')
    plt.colorbar()

    hori_idx = 650
    plt.sca(ax[2])
    # plt.plot(recon.as_array()[:,hori_idx]) # horizontal_x fixed
    plt.plot(recon_bhc.as_array()[hori_idx,:]) # horizontal_y fixed
    plt.xlabel('horizontal_x')
    plt.title(f'Intensity profile for horizontal_y={hori_idx} on the reconstruction')
    plt.grid(True)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    # plt.savefig(os.path.join(base_dir, 'plots/t_bhc_shapes.pdf'))
    plt.show()


######

def X20_raw_proj():
    # im1 = io.imread('plots/X20_raw_0785.png')
    # im2 = io.imread('plots/X20_raw_1193.png')
    im1 = np.load('plots/X20_raw_0785.npy')
    im2 = np.load('plots/X20_raw_1193.npy')

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    cax = ax[0].imshow(im1, cmap='grey')
    # fig.colorbar(cax, ax=ax[0])
    ax[0].set_title("X20_0785.tif")
    ax[1].imshow(im2, cmap='grey')
    ax[1].set_title("X20_1193.tif")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'plots/X20_raw_proj.pdf'), format='pdf', dpi=300)
    plt.show()


###
def clip_otsu_segment(recon, ig, clip=True):
    if clip:
        tau = threshold_otsu(np.clip(recon,a_min=0,a_max=None))
    else:
        tau = threshold_otsu(recon)
    
    segmented = ImageData(array=np.array(recon > tau, dtype='float32'), geometry=ig)
    return segmented
def compare_otsu_segmentation():
    data = load_centre('X20_cor.pkl')
    # data = load_centre('X16_cor.pkl')
    ag = data.geometry
    ig = ag.get_ImageGeometry()
    A = ProjectionOperator(ig, ag, direct_method='Siddon', device='gpu')
    recon = FDK(data).run()

    figsize = [(14,6), (12,5)][1]
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    otsu_seg = clip_otsu_segment(recon.as_array(), ig, clip=0)
    # otsu_abs = clip_otsu_segment(np.abs(recon.as_array()), ig, clip=0)
    ax[0].set_xlabel('horizontal_x')
    ax[0].set_ylabel('horizontal_y')
    ax[0].imshow(otsu_seg.as_array(), origin='lower', cmap='gray')
    ax[0].set_title('Otsu segmentation on initial FDK center slice')
    
    
    path_lengths = A.direct(otsu_seg)
    mask = (path_lengths.as_array() > 0) & (data.as_array() > 0.25)
    x = np.array(path_lengths.as_array()[mask])
    y = np.array(data.as_array()[mask])
    norms = {'lin': Normalize(), 'log': LogNorm()}
    color_norm = 'log'
    counts, x_edges, y_edges, _ = plt.hist2d(x,y,bins=100, range=[[0,x.max()],[0,y.max()]], norm=norms[color_norm])

    plt.sca(ax[1])
    plt.xlabel('Path length')
    plt.ylabel('Absorption')
    plt.title('Filtered 2D histogram')
    plt.colorbar()
    plt.tight_layout()
    # plt.subplots_adjust(top=0.85)
    plt.savefig(os.path.join(base_dir, 'plots/X20_initial_otsu_2Dhist.pdf'))
    plt.show()
    

    plt.figure()
    plt.hist(recon.as_array().flatten(),bins=50)
    plt.title('Histogram of initial FDK center slice')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(os.path.join(base_dir, 'plots/X20_initial_hist.pdf'))
    plt.show()


def compare_BHC_fits():
    from matplotlib import rc
    plt.rcParams.update({'font.size': 14})
    rc('text', usetex=True)
    rc('font', family='serif')

    data = load_centre('X20_cor.pkl')
    # data = load_centre('X16_cor.pkl')
    ag = data.geometry
    ig = ag.get_ImageGeometry()
    A = ProjectionOperator(ig, ag, direct_method='Siddon', device='gpu')
    recon = FDK(data).run()
    segmentation = clip_otsu_segment(recon.as_array(), ig, clip=0)
    path_lengths = A.direct(segmentation)

    mask = (path_lengths.as_array() > 0.05) & (data.as_array() > 0.25)
    def f_poly1(x, *a):
        return a[0]*x**3
    def f_poly2(x, *a):
        return a[0]*x**5
    
    shift = 0.05
    const = np.log10(shift)
    def f_poly3(x, *a):
        # return a[0]*np.log10(x+0.1) + 1
        # return 10**((x-1)/a[0])-0.1
        return 10**((x+const)/a[0]) - shift

    bhc = BHC(path_lengths, data, None, f_poly1, num_bins=100, mask=mask, n_poly=1)
    bhc.get_hist_fit_plot()
    bhc.plot_fits(show_hist=False, make_trans_plot=False, label=r'$cx^3$', linewidth=3, color='red')
    print(bhc.popt_poly)
    
    bhc.f_poly = f_poly2
    bhc.perform_fit()
    bhc.plot_fits(show_hist=False, make_trans_plot=False, label=r'$cx^5$', linewidth=3, color='magenta')
    print(bhc.popt_poly)

    bhc.f_poly = f_poly3
    # bhc.n_poly = 2
    bhc.perform_fit()
    bhc.plot_fits(show_hist=False, make_trans_plot=False, label=rf'$10^{{(x+\log_{{10}}({shift}))/c}} - {shift}$', linewidth=3, color='black')
    print(bhc.popt_poly)

    plt.legend(loc='lower right')
    plt.title(r'Comparing different $f_p$ fits')
    # plt.savefig(os.path.join(base_dir, 'plots/X20_comparing_poly_fits.pdf'))
    plt.show()

    bhc.f_poly = f_poly1
    _,recon1 = bhc.run(verbose=0)

    bhc.f_poly = f_poly2
    _,recon2 = bhc.run(verbose=0)

    bhc.f_poly = f_poly3
    _,recon3 = bhc.run(verbose=0)
    
    # show2D(recon1)
    # show2D(recon2)
    # show2D(recon3)



    from matplotlib import rc
    plt.rcParams.update({'font.size': 12})
    rc('text', usetex=True)
    rc('font', family='serif')
    # figsize = [(14,6), (12,5)][1]
    figsize = [(9,8)][0]
    fig, ax = plt.subplots(2,1, figsize=figsize)

    hori_x_slice = slice(300,750)
    plt.sca(ax[0])
    plt.imshow(recon1.as_array()[hori_x_slice], origin='lower', cmap='gray')
    # plt.xlabel('horizontal_x')
    # plt.ylabel('horizontal_y')
    plt.title(r'FDK reconstruction from $f_p=cx^3$')
    plt.colorbar()

    plt.sca(ax[1])
    plt.imshow(recon2.as_array()[hori_x_slice], origin='lower', cmap='gray')
    # plt.xlabel('horizontal_x')
    # plt.ylabel('horizontal_y')
    plt.title(r'FDK reconstruction from $f_p=cx^5$')
    plt.colorbar()

    plt.tight_layout()
    # plt.savefig(os.path.join(base_dir, 'plots/X20_poly_recons.pdf'))
    plt.show()


    ############### 3 instead
    from matplotlib import rc
    plt.rcParams.update({'font.size': 12})
    rc('text', usetex=True)
    rc('font', family='serif')
    # figsize = [(14,6), (12,5)][1]
    figsize = [(9,12)][0]
    fig, ax = plt.subplots(3,1, figsize=figsize)

    kwargs = {}

    hori_x_slice = slice(300,750)
    plt.sca(ax[0])
    # plt.imshow(recon1.as_array()[hori_x_slice], origin='lower', cmap='gray')
    plt.imshow(recon1.as_array()[hori_x_slice], origin='lower', cmap='gray')
    # plt.xlabel('horizontal_x')
    # plt.ylabel('horizontal_y')
    plt.title(r'FDK reconstruction from $f_p=cx^3$')
    plt.colorbar()

    plt.sca(ax[1])
    plt.imshow(recon2.as_array()[hori_x_slice], origin='lower', cmap='gray')
    # plt.xlabel('horizontal_x')
    # plt.ylabel('horizontal_y')
    plt.title(r'FDK reconstruction from $f_p=cx^5$')
    plt.colorbar()

    plt.sca(ax[2])
    plt.imshow(recon3.as_array()[hori_x_slice], origin='lower', cmap='gray')
    # plt.xlabel('horizontal_x')
    # plt.ylabel('horizontal_y')
    plt.title(rf'FDK reconstruction from $f_p=10^{{(x+\log_{{10}}({shift}))/c}} - {shift}$')
    plt.colorbar()

    plt.tight_layout()
    # plt.savefig(os.path.join(base_dir, 'plots/X20_poly_recons.pdf'))
    plt.show()

    ########### with raw FDK as well
    from matplotlib import rc
    plt.rcParams.update({'font.size': 16})
    rc('text', usetex=True)
    rc('font', family='serif')
    # figsize = [(14,6), (12,5)][1]
    figsize = [(9,16)][0]
    fig, ax = plt.subplots(4,1, figsize=figsize)

    im_args = {
        'origin': 'lower', 'cmap': 'gray'
        , 'vmin': -0.4, 'vmax': 0.8 # comment this out if automatic scales should be used
    }

    hori_x_slice = slice(300,750)
    plt.sca(ax[0])
    # plt.imshow(recon.as_array()[hori_x_slice], origin='lower', cmap='gray')
    plt.imshow(recon.as_array()[hori_x_slice], **im_args)
    # plt.xlabel('horizontal_x')
    # plt.ylabel('horizontal_y')
    plt.title(r'Raw FDK reconstruction')
    plt.colorbar()

    plt.sca(ax[1])
    # plt.imshow(recon1.as_array()[hori_x_slice], origin='lower', cmap='gray')
    plt.imshow(recon1.as_array()[hori_x_slice], **im_args)
    # plt.xlabel('horizontal_x')
    # plt.ylabel('horizontal_y')
    plt.title(r'FDK reconstruction from $f_p=cx^3$')
    plt.colorbar()

    plt.sca(ax[2])
    plt.imshow(recon2.as_array()[hori_x_slice], **im_args)
    # plt.xlabel('horizontal_x')
    # plt.ylabel('horizontal_y')
    plt.title(r'FDK reconstruction from $f_p=cx^5$')
    plt.colorbar()

    plt.sca(ax[3])
    plt.imshow(recon3.as_array()[hori_x_slice], **im_args)
    # plt.xlabel('horizontal_x')
    # plt.ylabel('horizontal_y')
    plt.title(rf'FDK reconstruction from $f_p=10^{{(x+\log_{{10}}({shift}))/c}} - {shift}$')
    plt.colorbar()

    plt.tight_layout()
    # plt.savefig(os.path.join(base_dir, 'plots/X20_poly_recons.pdf'))
    plt.show()

def bhc_iter():
    from matplotlib import rc
    plt.rcParams.update({'font.size': 14})
    rc('text', usetex=True)
    rc('font', family='serif')

    data = load_centre('X20_cor.pkl')
    # data = load_centre('X16_cor.pkl')
    ag = data.geometry
    ig = ag.get_ImageGeometry()
    A = ProjectionOperator(ig, ag, direct_method='Siddon', device='gpu')

    y_slice = slice(300,750)
    # y_slice = slice(700,1300)

    def f_poly1(x, *a):
        return a[0]*x**3
    def f_poly2(x, *a):
        return a[0]*x**5
    
    shift = 0.05
    const = np.log10(shift)
    def f_poly3(x, *a):
        # return a[0]*np.log10(x+0.1) + 1
        # return 10**((x-1)/a[0])-0.1
        return 10**((x+const)/a[0]) - shift

    ###
    f_polys = [f_poly1,f_poly2]
    fig_seg,ax_seg = plt.subplots(2,1,figsize=(9,8))
    fig_fit,ax_fit = plt.subplots(1,2,figsize=(11,5))
    labels = [r'$f_p=cx^3$', r'$f_p=cx^5$']

    plt.figure()
    for j,f_poly in enumerate(f_polys):
        data_bhc = data
        bhcs = []
        recon_bhcs = []
        segmentations = []
        recon = FDK(data).run(verbose=0)
        popt_polys = []

        N = 3
        for i in range(N):
            print(f'Iteration {i}')
            segmentation = clip_otsu_segment(recon.as_array(), ig, clip=0)
            path_lengths = A.direct(segmentation)
            # mask = (path_lengths.as_array() > 0.05) & (data_bhc.as_array() > 0.25)
            mask = (path_lengths.as_array() > 0.05) & (data.as_array() > 0.25)
            bhc = BHC(path_lengths, data, None, f_poly, num_bins=100, mask=mask, n_poly=1)
            if i < (N-1):
                data_bhc,recon = bhc.run(verbose=1)
            else:
                plt.figure(fig_fit)
                plt.sca(ax_fit.flatten()[j])
                bhc.get_hist_fit_plot()
                bhc.plot_fits(show_hist=False, make_trans_plot=False, linewidth=3, color='red')
                plt.title(rf'Converged fit for {labels[j]}')
                bhc.perform_correction()
                plt.figure()
                data_bhc,recon_bhc = bhc.data_bhc,bhc.recon_bhc
            # print(f'popt_poly: {bhc.popt_poly}')
            # show2D(recon.as_array()[y_slice])
            show2D(segmentation.as_array()[y_slice])

            popt_polys.append(bhc.popt_poly)
            bhcs.append(bhc)
            recon_bhcs.append(recon)

            if i < (N-1):
                segmentations.append(segmentation)
            else:
                plt.figure(fig_seg)
                plt.sca(ax_seg.flatten()[j])
                plt.imshow(segmentation.as_array()[y_slice], origin='lower', cmap='gray')
                plt.title(rf'Segmentation for BHC with {labels[j]}')
                plt.figure()

        segmentation = clip_otsu_segment(recon.as_array(), ig, clip=0)
        print(f'popt_polys: {popt_polys}')

        plt.figure(fig_fit)
        plt.tight_layout()
        # plt.savefig(os.path.join(base_dir,'plots/X20_bhc_conv_fits.pdf'))
        plt.show()

        plt.figure(fig_seg)
        plt.tight_layout()
        # plt.savefig(os.path.join(base_dir,'plots/X20_bhc_conv_segm.pdf'))
        plt.show()

def photon_shapes():
    # data = load_centre('X20_cor.pkl')
    # ag = data.geometry
    # ig = ag.get_ImageGeometry()

    physical_size = 0.2
    # physical_size = 0.5
    # physical_size = 0.01
    voxel_num = 1000
    ag,ig = setup_generic_cil_geometry(physical_size=physical_size,voxel_num=voxel_num)

    A = ProjectionOperator(ig, ag, direct_method='Siddon', device='gpu')
    
    im_arr = io.imread(os.path.join(base_dir,'test_images/test_image_shapes3.png'))
    im_arr = color.rgb2gray(im_arr) > 0

    im = ImageData(array=im_arr.astype('float32'), geometry=ig)
    show2D(im)

    mu = fun_attenuation(plot=False)
    bin_centers, bin_heights = generate_spectrum(plot=False,filter=0.0)
    mu_eff = np.sum(bin_heights * mu(bin_centers))
    # mu_eff = 19
    d = A.direct(im)#/10000
    trans = np.exp(-d*mu_eff)
    I0 = 60000.0
    # I0 = 100.0
    I_noisy = np.random.poisson(lam=I0*trans.as_array())

    plt.plot(d.as_array().flatten(),I_noisy.flatten(),'.')
    plt.show()
    print(f'min count: {np.min(I_noisy)}')

    b = AcquisitionData(array=d*mu_eff, geometry=ag)
    eps = 0*1e-6
    b_noisy = TransmissionAbsorptionConverter()(AcquisitionData(array=np.array(I_noisy/I0+eps, dtype='float32'), geometry=ag))
    # b_noisy = AcquisitionData(array=np.array(-np.log(I_noisy/I0),dtype='float32'), geometry=ag)
    
    plt.plot(d.as_array().flatten(),b_noisy.as_array().flatten(),'.')
    plt.show()

    recon = FBP(b, image_geometry=ig).run(verbose=0)
    show2D(recon, title='recon')
    recon_noisy = FBP(b_noisy, image_geometry=ig).run(verbose=0)
    show2D(recon_noisy, title='recon_noisy')

    tau = -np.log(eps)*0.9
    tau = 7
    b_interp = lin_interp_sino2D(b_noisy, tau)
    recon_interp = FBP(b_interp, image_geometry=ig).run(verbose=0)
    show2D(recon_interp)

    plt.hist(recon.as_array().flatten(), bins=100)
    plt.show()
    plt.hist(recon_noisy.as_array().flatten(), bins=100)
    # plt.show()
    plt.hist(recon_interp.as_array().flatten(), bins=100)
    plt.show()

def photon_shapes_bhc():
    # data = load_centre('X20_cor.pkl')
    # ag = data.geometry
    # ig = ag.get_ImageGeometry()

    # physical_size = 0.2
    physical_size = 0.3
    # physical_size = 0.01
    voxel_num = 1000
    ag,ig = setup_generic_cil_geometry(physical_size=physical_size,voxel_num=voxel_num)

    A = ProjectionOperator(ig, ag, direct_method='Siddon', device='gpu')
    
    im_arr = io.imread(os.path.join(base_dir,'test_images/test_image_shapes3.png'))
    im_arr = color.rgb2gray(im_arr) > 0

    im = ImageData(array=im_arr.astype('float32'), geometry=ig)
    show2D(im)

    mu = fun_attenuation(plot=False)
    bin_centers, bin_heights = generate_spectrum(plot=False,filter=0.0)
    mu_eff = np.sum(bin_heights * mu(bin_centers))
    # mu_eff = 19
    d = A.direct(im)#/10000
    trans = np.exp(-d*mu_eff)
    # I0 = 1e5
    I0 = 60000.0
    # I0 = 100.0
    I_noisy = np.random.poisson(lam=I0*trans.as_array())

    plt.plot(d.as_array().flatten(),I_noisy.flatten(),'.')
    plt.show()
    print(f'min count: {np.min(I_noisy)}')

    b = AcquisitionData(array=d*mu_eff, geometry=ag)
    eps = 1e-7
    # eps = 1e-5
    b_noisy = TransmissionAbsorptionConverter()(AcquisitionData(array=np.array(I_noisy/I0+eps, dtype='float32'), geometry=ag))
    # b_noisy = AcquisitionData(array=np.array(-np.log(I_noisy/I0),dtype='float32'), geometry=ag)
    
    fig,ax = plt.subplots(figsize=(10,6))
    plt.sca(ax)
    plt.plot(d.as_array().flatten(),b_noisy.as_array().flatten(),'k.', alpha=0.3, markersize=3)
    plt.title('Noisy absorption data as function of ground truth path lengths')
    plt.grid(True)
    plt.xlabel('mm')
    plt.savefig(os.path.join(base_dir, 'plots/shapes_mono_noisy_abs.png'))
    plt.show()

    # recon = FBP(b, image_geometry=ig).run(verbose=0)
    # show2D(recon, title='recon')
    recon_noisy = FBP(b_noisy, image_geometry=ig).run(verbose=0)
    show2D(recon_noisy, title='FBP reconstruction of noisy data', fix_range=(-75,150))

    tau = -np.log(eps)*0.9
    # tau = 7
    tau = 10
    b_interp = lin_interp_sino2D(b_noisy, tau)
    recon_interp = FBP(b_interp, image_geometry=ig).run(verbose=0)
    show2D(recon_interp, title='FBP reconstruction of interpolated noisy data')

    ######
    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    
    im_args = {
        'cmap': 'gray', 'origin': 'lower'
        , 'vmin': -75.0, 'vmax': 150.0
    }
    im0 = ax[0].imshow(recon_noisy.as_array(), **im_args)
    ax[0].set_title('Noisy data')
    fig.colorbar(im0, ax=ax[0], location='bottom')
    
    im1 = ax[1].imshow(recon_interp.as_array(), **im_args)
    ax[1].set_title('Interpolated noisy data')
    fig.colorbar(im1, ax=ax[1], location='bottom')

    fig.suptitle('FBP reconstructions', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'plots/shapes_mono_recons.pdf'))
    plt.show()
    #########


    # plt.hist(recon.as_array().flatten(), bins=100)
    # plt.show()

    fig,ax = plt.subplots(figsize=(10,6))
    plt.sca(ax)
    plt.hist(recon_noisy.as_array().flatten(), bins=200, label='Noisy data')
    # plt.show()
    plt.hist(recon_interp.as_array().flatten(), bins=100, label='Interpolated noisy data',alpha=0.7,color='red')
    plt.yscale('log')
    plt.title('Histograms of FBP reconstructions')
    plt.xlabel('Intensity')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(base_dir, 'plots/shapes_mono_hists.pdf'))
    plt.show()

    tau_noisy = threshold_otsu(recon_noisy.as_array())
    tau_interp = threshold_otsu(recon_interp.as_array())
    print(f'tau_noisy, tau_interp = {tau_noisy}, {tau_interp}')

    segm_noisy = recon_noisy > tau_noisy
    segm_interp = recon_interp > tau_interp
    # show2D([segm_noisy,segm_interp], title=['title1','title2'])

    #########
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].imshow(segm_noisy, cmap='gray', origin='lower')
    ax[0].set_title('Noisy data')
    
    ax[1].imshow(segm_interp, cmap='gray', origin='lower')
    ax[1].set_title('Interpolated noisy data')
    fig.suptitle('Otsu segmentations', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'plots/shapes_mono_segms.pdf'))
    plt.show()
    ########


def test_linear_interpolation():
    from matplotlib import rc
    plt.rcParams.update({'font.size': 14})
    rc('text', usetex=True)
    rc('font', family='serif')
    def compare_lin_interp(ang_idx,indices=None,save=False):
        n_rays = data.shape[1]
        if indices is None:
            indices = np.arange(0,n_rays)
        plt.figure(figsize=(10,8))
        plt.title(f'Single projection at angle_index={ang_idx}')
        plt.xlabel('Panel index')
        plt.ylabel('Absorption')
        plt.plot([indices.min(),indices.max()],[tau,tau],label=r'$\tau$')
        plt.plot(indices,data.as_array()[ang_idx,indices], label='Raw data', marker='.', markersize=8)
        plt.plot(indices,data_interp.as_array()[ang_idx,indices], label='Interpolated data')
        plt.legend()
        plt.grid(True)
        if save:
            plt.savefig(os.path.join(base_dir, 'plots/lin_interp_example.pdf'))
        plt.show()

    file_path = os.path.join(base_dir,'centres/X20_cor.pkl')
    # file_path = os.path.join(base_dir,'centres/X16_cor.pkl')
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    data_trans = AbsorptionTransmissionConverter()(data)
    show2D(data_trans, cmap='nipy_spectral', fix_range=(data_trans.min(),0.5))

    tau = 2.0
    # tau = -np.log(0.115)
    # tau = -np.log(0.13)
    # tau = -np.log(0.15)
    # tau = -np.log(0.07)
    # show1D(data_trans, slice_list=[('angle', 480)])
    # show1D(data, slice_list=[('angle', 800)])

    data_interp = lin_interp_sino2D(data, tau)
    ang_idx = 480
    # show1D([data,data_interp], slice_list=[[('angle', ang_idx)],[('angle', ang_idx)]], line_styles=['-','-'])
    indices = np.arange(275,750)
    # compare_lin_interp(ang_idx,indices=indices)
    # compare_lin_interp(530,indices=indices)
    # compare_lin_interp(530,indices=np.arange(600,700),save=True)

    recon = FDK(data).run(verbose=0)
    recon_interp = FDK(data_interp).run(verbose=0)
    show2D([recon,recon_interp],title=f'recon vs recon_interp for tau={tau}',fix_range=(-0.15,0.6))
    show2D([recon,recon_interp],title=f'recon vs recon_interp for tau={tau}')

    show2D(AbsorptionTransmissionConverter()(data_interp), cmap='nipy_spectral', fix_range=(data_trans.min(),0.5))

    ##########
    from matplotlib import rc
    plt.rcParams.update({'font.size': 14})
    rc('text', usetex=True)
    rc('font', family='serif')
    recon = FDK(data).run(verbose=0)
    # fig,ax = plt.subplots(4,1,figsize=(9,16))
    fig,ax = plt.subplots(2,1,figsize=(9,8))
    # fig,ax = plt.subplots(2,2,figsize=(14,8))
    hori_x_slice = slice(300,750)
    fix_range = (-0.15,0.6)

    plt.sca(ax.flatten()[0])
    plt.imshow(recon.as_array()[hori_x_slice],origin='lower',cmap='gray',vmin=fix_range[0],vmax=fix_range[1])
    plt.colorbar()
    plt.title('Raw data')

    # for i,tau in enumerate([2.2,2.0,1.9]):
    for i,tau in enumerate([1.9]):
        data_interp = lin_interp_sino2D(data, tau)
        recon_interp = FDK(data_interp).run(verbose=0)
        plt.sca(ax.flatten()[i+1])
        plt.imshow(recon_interp.as_array()[hori_x_slice],origin='lower',cmap='gray',vmin=fix_range[0],vmax=fix_range[1])
        plt.colorbar()
        plt.title(rf'Linearly interpolated with $\tau = {tau}$')
    
    # plt.suptitle('FDK reconstructions', fontsize=24)
    plt.tight_layout()
    # plt.savefig(os.path.join(base_dir, 'plots/X20_interp.pdf'))
    plt.show()

def single_projections():
    def show_proj(ang_idx,indices=None,save=False):
        n_rays = data.shape[1]
        if indices is None:
            indices = np.arange(0,n_rays)
        plt.figure(figsize=(12,6))
        plt.title(f'Single projection at angle_index={ang_idx}')
        plt.xlabel('Panel index')
        plt.ylabel('Absorption')
        plt.plot(indices,data.as_array()[ang_idx,indices])#, marker='.', markersize=8)
        # plt.legend()
        plt.grid(True)
        if save:
            plt.savefig(os.path.join(base_dir, 'plots/X20_single_projection.pdf'))
        plt.show()
    
    file_path = os.path.join(base_dir,'centres/X20_cor.pkl')
    # file_path = os.path.join(base_dir,'centres/X16_cor.pkl')
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    show_proj(900)
    show_proj(480, np.arange(400,600), save=True)
    show_proj(300, np.arange(500,600))

def X16_embeddings():
    from matplotlib import rc
    plt.rcParams.update({'font.size': 14})
    rc('text', usetex=True)
    rc('font', family='serif')

    def normalize_angle(idx):
        angle = np.mod(ag.angles[idx], 360)
        if angle > 180:
            angle -= 360
        return angle
    data = load_centre('X16_cor.pkl')
    ag = data.geometry
    ig = ag.get_ImageGeometry()

    recon = FDK(data).run(verbose=0)
    hori_idx_slice = slice(750,1400)
    fig,ax = plt.subplots(figsize=(12,5)); plt.sca(ax)
    plt.imshow(recon.as_array()[hori_idx_slice], cmap='gray')
    plt.title('FDK reconstruction of X16 center slice')
    plt.colorbar(location='bottom')
    plt.tight_layout()
    # plt.savefig(os.path.join(base_dir, 'plots/X16_center.pdf'))
    plt.show()

    detector_num_pixels = data.get_dimension_size('horizontal')

    chosen_angle = -3.2
    idx_chosen_angle = np.argmin(np.mod(ag.angles - chosen_angle, 360))
    print(np.mod(ag.angles[idx_chosen_angle], 360))

    # horizontal_idxs = np.round(np.linspace(0,detector_num_pixels-1,20)).astype(int)
    horizontal_idxs = np.round(np.linspace(0,detector_num_pixels-1,100)).astype(int)
    # horizontal_idxs = np.round(np.linspace(800,1200,20)).astype(int)
    # make_projection_plot(data,recon,idx_chosen_angle,horizontal_idxs)
    make_projection_plot(
        data,recon,idx_chosen_angle,horizontal_idxs,scale_abs=0.1,
        scale_proj_axis=0.13,ray_opacity=0.7,ray_thickness=1,scale_factor=1.0,
        scale_factor_y=0.5, lims=[(0,2000), (400,1600)], show=False
    )
    plt.title(rf'Projection at angle ${normalize_angle(idx_chosen_angle):.2f}^\circ$')
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    plt.savefig(os.path.join(base_dir, 'plots/X16_embed_short_side.pdf'))
    plt.show()

    chosen_angle = -3.2+90
    # chosen_angle = -3.2
    idx_chosen_angle = np.argmin(np.mod(ag.angles - chosen_angle, 360))
    print(np.mod(ag.angles[idx_chosen_angle], 360))
    horizontal_idxs = np.round(np.linspace(0,detector_num_pixels-1,100)).astype(int)
    horizontal_idxs = np.arange(1010-10,1125+10,3)
    make_projection_plot(
        data,recon,idx_chosen_angle,horizontal_idxs,scale_abs=0.08,
        scale_proj_axis=0.85,ray_opacity=0.3,ray_thickness=0.3,scale_factor=1.1,
        scale_factor_y=0.4, lims=[(0,2200),(750,1500)], show=False
    )
    plt.title(rf'Projection at angle ${normalize_angle(idx_chosen_angle):.2f}^\circ$')
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(10, 4)
    plt.savefig(os.path.join(base_dir, 'plots/X16_embed_long_side.pdf'))
    plt.show()
    ###
    # chosen_angle = -3.2+90+180 -0
    # chosen_angle = -3.2+70
    chosen_angle = -3.2+60+180
    idx_chosen_angle = np.argmin(np.mod(ag.angles - chosen_angle, 360))
    print(np.mod(ag.angles[idx_chosen_angle], 360))
    horizontal_idxs = np.round(np.linspace(0,detector_num_pixels-1,500)).astype(int)
    # horizontal_idxs = np.arange(1010,1125)
    make_projection_plot(
        data,recon,idx_chosen_angle,horizontal_idxs,scale_abs=0.08,
        scale_proj_axis=0.85,ray_opacity=0.3,ray_thickness=0.3,scale_factor=1.3
    )

    make_projection_plot(
        data,recon,idx_chosen_angle,horizontal_idxs,scale_abs=0.14,
        scale_proj_axis=0.85,ray_opacity=0.3,ray_thickness=0.3,scale_factor=1.3
    )

    show1D(data,slice_list=[('angle', 300)])

def X20_embeddings():
    def normalize_angle(idx):
        angle = np.mod(ag.angles[idx], 360)
        if angle > 180:
            angle -= 360
        return angle

    data = load_centre('X20_cor.pkl')
    ag = data.geometry
    ig = ag.get_ImageGeometry()

    recon = FDK(data).run(verbose=0)
    detector_num_pixels = data.get_dimension_size('horizontal')

    chosen_angle = 0
    idx_chosen_angle = np.argmin(np.mod(ag.angles - chosen_angle, 360))
    print(np.mod(ag.angles[idx_chosen_angle], 360))
    horizontal_idxs = np.round(np.linspace(0,detector_num_pixels-1,300)).astype(int)
    # horizontal_idxs = np.arange(300,700)
    # horizontal_idxs = np.arange(498,500)
    # horizontal_idxs = [498,499,500,501,502]
    make_projection_plot(
        data,recon,idx_chosen_angle,horizontal_idxs,scale_abs=0.2,
        scale_proj_axis=0.3,ray_opacity=0.3,ray_thickness=1,scale_factor=0.9)
    
    ###
    chosen_angle = 30
    idx_chosen_angle = np.argmin(np.mod(ag.angles - chosen_angle, 360))
    print(np.mod(ag.angles[idx_chosen_angle], 360))
    make_projection_plot(
        data,recon,idx_chosen_angle,horizontal_idxs,scale_abs=0.2,
        scale_proj_axis=0.3,ray_opacity=0.3,ray_thickness=1,scale_factor=1.1)
    
    ###
    chosen_angle = 180
    idx_chosen_angle = np.argmin(np.mod(ag.angles - chosen_angle, 360))
    print(np.mod(ag.angles[idx_chosen_angle], 360))
    horizontal_idxs = np.arange(0,700)
    make_projection_plot(
        data,recon,idx_chosen_angle,horizontal_idxs,scale_abs=0.08,
        scale_proj_axis=0.08,ray_opacity=0.2,ray_thickness=1,
        lims=[(300,600),(350,550)],show=False)
    plt.title(rf'Projection at angle ${normalize_angle(idx_chosen_angle):.2f}^\circ$')
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(10, 7)
    plt.savefig(os.path.join(base_dir, 'plots/X20_embed_no_holes.pdf'))
    plt.show()
    ###
    idx_chosen_angle = 500
    horizontal_idxs = np.arange(590,750)
    make_projection_plot(
        data,recon,idx_chosen_angle,horizontal_idxs,scale_abs=0.05,
        scale_proj_axis=-0.3,ray_opacity=0.2,ray_thickness=1,lims=[(0,400),(400,650)])
    
    ###
    chosen_angle = 120+10
    idx_chosen_angle = np.argmin(np.mod(ag.angles - chosen_angle, 360))
    print(np.mod(ag.angles[idx_chosen_angle], 360))
    horizontal_idxs = np.arange(0,1000,3)
    make_projection_plot(
        data,recon,idx_chosen_angle,horizontal_idxs,scale_abs=0.08,
        scale_proj_axis=0.73,ray_opacity=0.2,ray_thickness=1,lims=[(100,1200),(-200,700)])
    
    ###
    chosen_angle = 95
    idx_chosen_angle = np.argmin(np.mod(ag.angles - chosen_angle, 360))
    print(np.mod(ag.angles[idx_chosen_angle], 360))
    horizontal_idxs = np.arange(0,1000,3)
    make_projection_plot(
        data,recon,idx_chosen_angle,horizontal_idxs,scale_abs=0.08,
        scale_proj_axis=0.83,ray_opacity=0.2,ray_thickness=1,
        show=False,lims=[(100,1100),(350,750)])
    plt.title(rf'Projection at angle ${normalize_angle(idx_chosen_angle):.2f}^\circ$')
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(10, 4)
    # plt.savefig(os.path.join(base_dir, 'plots/X20_embed_thick.pdf'))
    plt.show()

def X12_X22_embeddings():
    def normalize_angle(idx):
        angle = np.mod(ag.angles[idx], 360)
        if angle > 180:
            angle -= 360
        return angle

    data = load_centre('X12-X22_cor.pkl')
    ag = data.geometry
    ig = ag.get_ImageGeometry()

    recon = FDK(data).run(verbose=0)
    detector_num_pixels = data.get_dimension_size('horizontal')
    
    chosen_angle = [-20,45][1]
    idx_chosen_angle = np.argmin(np.mod(ag.angles - chosen_angle, 360))
    horizontal_idxs = np.arange(0,1000,3)
    make_projection_plot(
        data,recon,idx_chosen_angle,horizontal_idxs,scale_abs=0.23,
        scale_proj_axis=0.65,ray_opacity=0.3,ray_thickness=1,scale_factor=0.9,
        show=False,lims=[(100,1100),(400,1200)])
    plt.title(rf'Projection at angle ${normalize_angle(idx_chosen_angle):.2f}^\circ$')
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.savefig(os.path.join(base_dir, 'plots/X12-X22_embed_scatter.pdf'))
    plt.show()

def X20_reg():
    data = load_centre('X20_cor.pkl')
    ag = data.geometry
    ig = ag.get_ImageGeometry()
    A = ProjectionOperator(ig, ag, direct_method='Siddon', device='gpu')

    data_pad = cil.processors.Padder(pad_width=300)(data)
    ag_pad = data_pad.geometry
    ig_pad = ag_pad.get_ImageGeometry()
    A_pad = ProjectionOperator(ig_pad, ag_pad, direct_method='Siddon', device='gpu')

    recon = FDK(data).run(verbose=0)
    recon_pad = FDK(data_pad).run(verbose=0)
    # show2D(recon)
    # show2D(recon_pad)

    hori_y_slice = slice(300,750)
    #####
    update_interval = 1
    F = LeastSquares(A, data)
    G = IndicatorBox(lower=0.0)
    x0 = ig.allocate(0.0)
    fista_NN = FISTA(f=F, g=G, initial=x0,
                     max_iteration=1000,
                     update_objective_interval=update_interval)
    fista_NN.run(50, verbose=2)
    # show2D(fista_NN.solution.as_array()[hori_y_slice])


    #####
    from matplotlib import rc
    plt.rcParams.update({'font.size': 14})
    rc('text', usetex=True)
    rc('font', family='serif')

    alphas = [1,10,100]
    objectives = []
    solutions = []
    for i,alpha in enumerate(alphas):
        F = LeastSquares(A, data)
        # F = LeastSquares(A_pad, data_pad)
        G = FGP_TV(alpha=alpha, nonnegativity=True, device='gpu')
        # G = TotalVariation()
        # x0 = ig.allocate(0.0)
        x0 = recon
        # x0 = recon_pad
        update_interval = 1
        fista_TV = FISTA(initial=x0, f=F, g=G,
                        max_iteration=1000, update_objective_interval=update_interval)
        fista_TV.run(50, verbose=2)
        # show2D(fista_TV.solution.as_array()[hori_y_slice], title=f'alpha = {alpha}')
        solutions.append(fista_TV.solution.as_array()[hori_y_slice])
        objectives.append(fista_TV.objective)

    alphas.insert(0,0)
    solutions.insert(0,fista_NN.solution.as_array()[hori_y_slice])
    objectives.insert(0,fista_NN.objective)

    N = len(alphas)
    fig,ax = plt.subplots(N,1,figsize=(9,16))
    im_args = {
        'origin': 'lower', 'cmap': 'gray'
        , 'vmin': 0.0, 'vmax': 0.7 # comment this out if automatic scales should be used
    }
    for i in range(N):
        plt.sca(ax.flatten()[i])
        # plt.imshow(solutions[i], origin='lower', cmap='gray')
        plt.imshow(solutions[i], **im_args)
        plt.title(rf'NN+TV with $\alpha = {alphas[i]}$')
        plt.colorbar()
    plt.tight_layout()
    # plt.savefig(os.path.join(base_dir, 'plots/X20_TV_recons.pdf'))
    plt.show()

    fig, ax = plt.subplots(figsize=(8,4))
    plt.sca(ax)
    for i in range(N):
        # plt.plot(objectives[i], label=rf'$\alpha = {alphas[i]}$')
        plt.plot(range(1,len(objectives[i])),objectives[i][1:], label=rf'$\alpha = {alphas[i]}$')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.title('Convergence plot of the objective function for NN+TV')
    plt.xlabel('Iteration')
    plt.ylabel('Objective value')
    plt.tight_layout()
    # plt.savefig(os.path.join(base_dir, 'plots/X20_TV_conv.pdf'))
    plt.show()

def X20_bhc_reg():
    from matplotlib import rc
    plt.rcParams.update({'font.size': 14})
    rc('text', usetex=True)
    rc('font', family='serif')

    data = load_centre('X20_cor.pkl')
    # data = load_centre('X16_cor.pkl')
    ag = data.geometry
    ig = ag.get_ImageGeometry()
    A = ProjectionOperator(ig, ag, direct_method='Siddon', device='gpu')
    recon = FDK(data).run(verbose=0)

    ########### NN
    F = LeastSquares(A, data)
    G = IndicatorBox(lower=0.0)
    # x0 = ig.allocate(0.1)
    x0 = recon
    NN = FISTA(f=F, g=G, initial=x0, 
                    max_iteration=1000,
                    update_objective_interval=10)
    NN.run(100)
    ################### bhc1NN
    def f_poly1(x, *a):
        return a[0]*x**3
    def f_poly2(x, *a):
        return a[0]*x**5
    
    recon_bhc = FDK(data).run()
    for i in range(2):
        segmentation = clip_otsu_segment(recon_bhc.as_array(), ig, clip=0)
        path_lengths = A.direct(segmentation)
        mask = (path_lengths.as_array() > 0.05) & (data.as_array() > 0.25)
        bhc = BHC(path_lengths, data, None, f_poly1, num_bins=100, mask=mask, n_poly=1)
        data_bhc,recon_bhc = bhc.run(verbose=0)

    ############# 
    F = LeastSquares(A, data_bhc)
    G = IndicatorBox(lower=0.0)
    # x0 = ig.allocate(0.1)
    x0 = recon
    bhc1NN = FISTA(f=F, g=G, initial=x0, 
                    max_iteration=1000,
                    update_objective_interval=10)
    bhc1NN.run(100)
    show2D(bhc1NN.solution, fix_range=(0.0,0.4))
    tau = threshold_otsu(bhc1NN.solution.as_array())
    show2D(bhc1NN.solution > tau)

    show2D([NN.solution,bhc1NN.solution], fix_range=(0.0,0.4))

    ############ bhc2NN
    recon_bhc = FDK(data).run()
    for i in range(2):
        segmentation = clip_otsu_segment(recon_bhc.as_array(), ig, clip=0)
        path_lengths = A.direct(segmentation)
        mask = (path_lengths.as_array() > 0.05) & (data.as_array() > 0.25)
        bhc = BHC(path_lengths, data, None, f_poly2, num_bins=100, mask=mask, n_poly=1)
        data_bhc,recon_bhc = bhc.run(verbose=0)

    F = LeastSquares(A, data_bhc)
    G = IndicatorBox(lower=0.0)
    # x0 = ig.allocate(0.1)
    x0 = recon
    bhc2NN = FISTA(f=F, g=G, initial=x0, 
                    max_iteration=1000,
                    update_objective_interval=10)
    bhc2NN.run(100)
    show2D(bhc2NN.solution, fix_range=(0.0,0.4))
    tau = threshold_otsu(bhc2NN.solution.as_array())
    show2D(bhc2NN.solution > tau)

    show2D([NN.solution,bhc2NN.solution], fix_range=(0.0,0.4))
    #################
    y_slice = slice(300,750)
    im_args = {
        'origin': 'lower', 'cmap': 'gray'
        , 'vmin': 0.0, 'vmax': 0.4
    }
    fig,ax = plt.subplots(3,1,figsize=(9,12))

    plt.sca(ax[0])
    plt.imshow(NN.solution.as_array()[y_slice], **im_args)
    plt.colorbar()
    plt.title(rf'Raw data and NN constraints')

    plt.sca(ax[1])
    plt.imshow(bhc1NN.solution.as_array()[y_slice], **im_args)
    plt.colorbar()
    plt.title(rf'BHC with $f_p=cx^3$ and NN constraints')

    plt.sca(ax[2])
    plt.imshow(bhc2NN.solution.as_array()[y_slice], **im_args)
    plt.colorbar()
    plt.title(rf'BHC with $f_p=cx^5$ and NN constraints')

    plt.tight_layout()
    # plt.savefig(os.path.join(base_dir, 'plots/X20_bhc_NN.pdf'))
    plt.show()



def backprojection_mask():
    file_path = os.path.join(base_dir,'centres/X20_cor.pkl')
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    ag = data.geometry
    ig = ag.get_ImageGeometry()
    tau = 0.3
    air_proj_mask = 1000*np.array(data.as_array() < tau, dtype=np.float32)
    # air_proj_mask = 1e6*np.array(data.as_array() < tau, dtype=np.float32)
    air_proj_mask = AcquisitionData(array=air_proj_mask, geometry=ag)

    fft_order = 11
    recon_air = FDK(air_proj_mask, filter=np.ones(2**fft_order,dtype=np.float32)).run(verbose=0)
    epsilon = 0.1
    recon_mask = recon_air < epsilon
    show2D(recon_air, fix_range=(0,10), cmap='nipy_spectral')
    show2D(recon_mask)

    fig,ax = plt.subplots(figsize=(9,4))
    plt.imshow(recon_mask[slice(300,750)], origin='lower', cmap='gray')
    plt.colorbar()
    plt.title('Mask from air ray back-projection')
    plt.tight_layout()
    # plt.savefig(os.path.join(base_dir,'plots/X20_air_mask.pdf'))
    plt.show()

    A = ProjectionOperator(ig, ag, 'Siddon', device='gpu')
    F = LeastSquares(A, data)
    x0 = ig.allocate(0.0)
    # x0 = FDK(data).run(verbose=0)
    # show2D(x0)

    lb = np.zeros(recon_mask.shape)
    lb[recon_mask] = -np.inf
    ub = np.zeros(recon_mask.shape)
    ub[recon_mask] = np.inf

    G = IndicatorBox(lower=lb, upper=ub)
    fista_air = FISTA(f=F, g=G, initial=x0, 
                      max_iteration=1000,
                      update_objective_interval=10)
    fista_air.run(50, verbose=1)
    show2D(fista_air.solution,'Masking')
    show2D(fista_air.solution,fix_range=(0,1.25))

    G = IndicatorBox(lower=0, upper=ub)
    fista_air_NN = FISTA(f=F, g=G, initial=x0, 
                         max_iteration=1000,
                         update_objective_interval=10)
    fista_air_NN.run(100, verbose=1)
    # goes back up at around 150
    
    fig,ax = plt.subplots(figsize=(9,4))
    plt.imshow(fista_air_NN.solution.as_array()[slice(300,750)], origin='lower', cmap='gray', vmax=0.8)
    plt.colorbar()
    plt.title('Masking with NN')
    plt.tight_layout()
    # plt.savefig(os.path.join(base_dir,'plots/X20_air_mask_recon.pdf'))
    plt.show()

    show2D(fista_air_NN.solution,'Masking with NN',size=(4,10))
    show2D(fista_air_NN.solution,fix_range=(0,1.25))

def X20_cor():
    from matplotlib import rc
    plt.rcParams.update({'font.size': 12})
    rc('text', usetex=True)
    rc('font', family='serif')

    file_path = os.path.join(base_dir,'centres/X20.pkl')
    with open(file_path, 'rb') as file:
        data_no_cor = pickle.load(file)

    file_path = os.path.join(base_dir,'centres/X20_cor.pkl')
    with open(file_path, 'rb') as file:
        data_cor = pickle.load(file)
    
    data_no_cor = TransmissionAbsorptionConverter()(data_no_cor)
    ag_no_cor = data_no_cor.geometry
    ig_no_cor = ag_no_cor.get_ImageGeometry()

    ag_cor = data_cor.geometry
    ig_cor = ag_cor.get_ImageGeometry()

    recon_no_cor = FDK(data_no_cor, ig_no_cor).run(verbose=0)
    recon_cor = FDK(data_cor, ig_cor).run(verbose=0)

    y_slice = slice(300,750)
    fig,ax = plt.subplots(2,1,figsize=(9,8))
    plt.sca(ax[0])
    plt.imshow(recon_no_cor.as_array()[y_slice], origin='lower', cmap='gray')
    plt.title('Without COR correction')
    plt.colorbar()

    plt.sca(ax[1])
    plt.imshow(recon_cor.as_array()[y_slice], origin='lower', cmap='gray')
    plt.title('With COR correction')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'plots/X20_cor_vs_no_cor.pdf'))
    plt.show()

    show2D(data_cor, size=(9,13)).save(os.path.join(base_dir, 'plots/X20_sino.pdf'))
    # fig = show2D(data_cor, size=(9,13)).figure
    # plt.figure(fig)
    # plt.title('Sinogram of X20 center slice')
    # plt.tight_layout()
    # plt.savefig(os.path.join(base_dir, 'plots/X20_sino.pdf'))
    # plt.show()
    

def X20_imitation():
    from matplotlib import rc
    plt.rcParams.update({'font.size': 12})
    rc('text', usetex=True)
    rc('font', family='serif')

    data = load_centre('X20_cor.pkl')
    ag = data.geometry
    ig = ag.get_ImageGeometry()
    A = ProjectionOperator(ig, ag, direct_method='Siddon', device='gpu')

    im = io.imread(os.path.join(base_dir,'test_images/X20_imi.png'))[::-1,:,0] > 0
    image = ImageData(array=im.astype('float32'), geometry=ig)
    
    y_slice = slice(300,750)
    fig,ax = plt.subplots(figsize=(9,4))
    plt.imshow(image.as_array()[y_slice],origin='lower',cmap='gray')
    plt.title('X20 imitation')
    plt.tight_layout()
    # plt.savefig(os.path.join(base_dir, 'plots/X20_imitation.pdf'))
    plt.show()

    mu = fun_attenuation(plot=False)
    bin_centers, bin_heights = generate_spectrum(plot=False,filter=0.00)
    num_bins = bin_centers.size

    d = A.direct(image)
    d = d.as_array()

    I = np.zeros(d.shape, dtype='float64')
    I_const = 0.08
    I0 = 0
    for i in range(num_bins):
        E = bin_centers[i]
        I0_E = bin_heights[i]
        I0 += I0_E
        I += I0_E * np.exp(-mu(E)*d)
        # I += I0_E * np.exp(-mu(E)*d) + I_const/num_bins
        # I += I0_E * np.exp(-mu(E)*d) + 5*1e-4

    I_P = I
    b_bh = AcquisitionData(array=np.array(-np.log(I_P/I0),dtype='float32'), geometry=ag)
    I = np.array(I_P + I_const,dtype='float32')
    
    b_bh_scatter = AcquisitionData(array=-np.log(I/I0), geometry=ag)
    # b = data.as_array()

    ##
    mu_eff = np.sum(bin_heights * mu(bin_centers))
    print(f'mu_eff: {mu_eff}')
    # b_mono = -np.log(np.exp(-mu_eff*d))
    b_mono = d*mu_eff
    # b_mono = AcquisitionData(array=np.exp(-b_mono), geometry=ag)
    b_mono = AcquisitionData(array=b_mono, geometry=ag)

### Reconstructions
    ## noise
    # cap = None
    # # cap = b.max()
    # b_noisy = skimage.util.random_noise(b.as_array(), mode='gaussian',clip=False, mean=0, var=1/(1e4*I))
    # b_noisy = np.clip(a=b_noisy, a_min=0, a_max=cap)
    # b_noisy = AcquisitionData(array=np.array(b_noisy,dtype='float32'), geometry=ag)

    # plt.scatter(A.direct(image).as_array().flatten(), b_noisy.as_array().flatten(), alpha=0.2, color='black', label='Observed',s=3)
    # plt.title('b_noisy vs true path lengths')
    # plt.xlabel('mm')
    
    # recon_noisy = FDK(b_noisy, image_geometry=ig).run(verbose=0)
    # show2D(recon_noisy, title='recon')

    # plt.plot(d.flatten(), b.as_array().flatten(), '.', alpha=0.5, color='black')
    # plt.grid(True)

    plt.rcParams.update({'font.size': 14})
    fig,ax = plt.subplots(2,1,figsize=(9,7))
    bins = [100,40]
    plt.sca(ax[0])
    x = d.flatten()
    y = b_bh.as_array().flatten()
    plt.hist2d(x, y, bins=bins, range=[[0, x.max()], [0, y.max()]], norm=LogNorm())
    plt.colorbar()
    plt.xlabel(r'Path length [mm]')
    plt.ylabel(r'Absorption')
    plt.title('2D histogram of polychromatic data with no scatter')

    plt.sca(ax[1])
    x = d.flatten()
    y = b_bh_scatter.as_array().flatten()
    plt.hist2d(x, y, bins=bins, range=[[0, x.max()], [0, y.max()]], norm=LogNorm())
    plt.colorbar()
    plt.xlabel(r'Path length [mm]')
    plt.ylabel(r'Absorption')
    plt.title('2D histogram of polychromatic data with constant scatter')

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'plots/X20_imi_2Dhist.pdf'))
    plt.show()

    recon_bh = FDK(b_bh, image_geometry=ig).run(verbose=0)
    recon_bh_scatter = FDK(b_bh_scatter, image_geometry=ig).run(verbose=0)
    # show2D(recon, title='recon')

    y_slice = slice(350,700)
    fig,ax = plt.subplots(2,1,figsize=(9,8))
    imshow_kwargs = {
        'origin': 'lower',
        'cmap': 'gray',
        'vmin': -1,
        'vmax': 2
    }

    plt.sca(ax[0])
    plt.imshow(recon_bh.as_array()[y_slice], **imshow_kwargs)
    plt.colorbar()
    plt.title('FDK of imitation with BH and no scatter')

    plt.sca(ax[1])
    plt.imshow(recon_bh_scatter.as_array()[y_slice], **imshow_kwargs)
    plt.colorbar()
    plt.title('FDK of imitation with BH and constant scatter')

    plt.tight_layout()
    # plt.savefig(os.path.join(base_dir, 'plots/X20_imi_recon.pdf'))
    plt.show()

def esc_shapes():
    from matplotlib import rc
    plt.rcParams.update({'font.size': 12})
    rc('text', usetex=True)
    rc('font', family='serif')
    
    data = load_centre('X20.pkl')
    ag = data.geometry
    ig = ag.get_ImageGeometry()

    filepath = os.path.join(base_dir,'test_images/test_image_shapes3.png')
    # filepath = os.path.join(base_dir,'test_images/esc_circles.png')
    # filepath = os.path.join(base_dir,'test_images/esc_geom.png')
    im_arr = io.imread(filepath)
    im_arr = color.rgb2gray(im_arr) > 0

    im = ImageData(array=im_arr.astype('float32'), geometry=ig)
    A = ProjectionOperator(ig, ag, 'Siddon', device='gpu')
    mu = 10 / (ig.voxel_num_x*ig.voxel_size_x) # scale to 1 mm
    # mu = 40 / (ig.voxel_num_x*ig.voxel_size_x)
    data = mu*A.direct(im)
    data.reorder('tigre')
    recon_P = FDK(data, image_geometry=ig).run(verbose=0)
    show2D(recon_P)

    ### Scatter simulation
    I_P = AbsorptionTransmissionConverter()(data).as_array()
    b = data.as_array()
    # factor = [5,20,40,100,200][2]
    # basis_params = [[0.15*1/factor,0,factor]]

    # factors = [40,100]
    factors = [60]
    basis_params = [[0.15*1/factor,0,factor] for factor in factors]

    I_S = compute_scatter_basis(b,basis_params)
    I_S_sum = np.sum(I_S,axis=0)/len(basis_params)
    # I_S = 0.05*I_S*3
    # I_S = 0.05*I_S*3
    I = I_P + I_S_sum

    basis_idx = 0
    idx = 0

    ########################
    fig,ax = plt.subplots(figsize=(8,5))
    plt.sca(ax)
    plt.plot(I_P[idx,:],label=r'$I_P$')
    # plt.plot(I_S[basis_idx,idx,:],label=r'$I_S$')
    plt.plot(I_S_sum[idx,:],label=r'$I_S$')
    # plt.plot(I[idx,:],label=r'I=I_P+I_S')
    plt.grid(True)
    plt.legend()
    plt.title('Projection with simulated scatter')
    # plt.savefig(os.path.join(base_dir, 'plots/shapes_scatter_proj.png'))
    plt.show()
    #####################

    data_scatter = TransmissionAbsorptionConverter()(AcquisitionData(array=np.array(I, dtype=np.float32), geometry=ag))
    recon = FDK(data_scatter, image_geometry=ig).run(verbose=0)

    ####################
    fig,ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(8,10))
    plt.sca(ax[0])
    im0 = plt.imshow(recon.as_array(), origin='lower', cmap='gray')#,vmin=-10, vmax=10)
    # plt.imshow(recon.as_array(), origin='lower', cmap='gray')
    plt.xlabel('horizontal_x')
    plt.ylabel('horizontal_y')
    plt.title(r'Reconstruction of $-\ln(I_P+I_S)$')
    plt.colorbar()
    # plt.colorbar(im0,fraction=0.046, pad=0.04)

    plt.sca(ax[1])
    hori_x,hori_y = 700,650

    plt.plot(recon.as_array()[:,hori_x]) # horizontal_x fixed
    plt.title(f'Intensity profile for horizontal_x={hori_x} on the reconstruction')
    plt.xlabel('horizontal_y')

    # plt.plot(recon.as_array()[hori_y,:]) # horizontal_y fixed
    # plt.title(f'Intensity profile for horizontal_y={hori_y} on the reconstruction')
    # plt.xlabel('horizontal_x')
    
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig(os.path.join(base_dir, 'plots/shapes_scatter_recon.pdf'))
    plt.show()
    ####################
    
    P = recon.as_array()
    b = data_scatter.as_array()
    ny,nx = ig.shape

    ### ESC step
    # basis_params = [[1/5,0,5],[1/10,0,10],[1/40,0,40]]

    # factor = [5,20,40,100,200][2]
    # basis_params = [[0.15*1/factor,0,factor]]
    # basis_params = [[0.15*1/factor,0,factor] for factor in factors]

    # factors = [40,100]
    factors = [60]
    basis_params = [[0.15*1/factor,0,factor] for factor in factors]

    trans = I
    nc = len(basis_params)

    basis_idx = 0
    if False:
        ### approximation (real world data)
        I_S = compute_scatter_basis(b,basis_params)
        I_S_sum = np.sum(I_S,axis=0)/len(basis_params)
        # I_S = 0.05*I_S
        # I_Q = trans-I_S
        I_Q = I - I_S_sum
        s = b[None,:,:] + np.log(I_Q)
        plt.plot(I_P[idx,:],label='I_P')
        plt.plot(I_Q[idx,:],label='I_Q')
        plt.plot(I[idx,:],label='I=I_P+I_S')
        plt.legend()
        plt.title(f'basis_idx: {basis_idx}')
        plt.grid(True)
        plt.show()
        ###
    else:
        ### what the model is based on
        I_S = compute_scatter_basis(-np.log(I_P),basis_params)
        I_S_sum = np.sum(I_S,axis=0)/len(basis_params)
        # I_S = 0.05*I_S
        # I_Q = trans-I_S
        I_Q = I - I_S_sum
        s = -np.log(I_S+I_P) + np.log(I_P)
        plt.plot(I_P[idx,:],label='I_P')
        plt.plot(I_Q[idx,:],label='I_Q')
        plt.plot(I[idx,:],label='I=I_P+I_S')
        plt.legend()
        plt.title(f'basis_idx: {basis_idx}')
        plt.grid(True)
        plt.show()
        ###

    # print(np.unravel_index(s.argmax(),s.shape), np.max(s))

    # fig,ax = plt.subplots(1,2,figsize=(10,5))
    # plt.title(f'basis_idx: {basis_idx}')
    # plt.sca(ax[0])
    # plt.plot(s[basis_idx,idx,:],label='s')
    # plt.legend()

    # plt.sca(ax[1])
    # plt.plot(-np.log(-s[basis_idx,idx,:]),label='-ln(-s)')
    # plt.legend()
    # plt.show()

    S = np.zeros((nc,*ig.shape))
    for i in range(nc):
        data_s_i = AcquisitionData(array=np.array(s[i], dtype='float32'), geometry=ag)
        S[i] = FDK(data_s_i).run(verbose=0).as_array()
        # show2D(-S[i])
        show2D(S[i],title=f'S_{i}')

    Mext = np.zeros((nx*ny,nc+1))
    Mext[:,0] = P.flatten()
    Mext[:,1:] = S.reshape(nc, nx*ny).T

    vg = VectorGeometry(length=nc+1)
    op1 = ESCMatrixOperator(Mext, domain_geometry=vg, range_geometry=ig)
    op2 = GradientOperator(domain_geometry=ig)
    K = CompositionOperator(op2,op1)
    F = MixedL21Norm()
    # F = MixedL11Norm()
    # F = SmoothMixedL21Norm(epsilon=1e-2)

    cext = VectorData(array=np.hstack((1,np.full(nc, 0.1, dtype=np.float32))), geometry=vg)

    l = -np.inf*np.ones(1+nc)
    l[0] = 1
    u = np.inf*np.ones(1+nc)
    u[0] = 1
    G = IndicatorBox(lower=l,upper=u)
    upd_int = 3
    pdhg = PDHG(f=F, g=G, operator=K, initial=cext, max_iteration=1000, update_objective_interval=3)
    # pdhg.run(iterations=150)
    pdhg.run(iterations=34)

    print(pdhg.solution.as_array())
    Q = op1.direct(pdhg.solution)
    show2D(Q, title=f"c_ext={pdhg.solution.as_array()}")
    show1D(Q, [('horizontal_x',hori_x)], title=f'horizontal_x={hori_x}', size=(8,3))
    # show1D(Q, [('horizontal_y',hori_y)], title=f'horizontal_y={hori_y}', size=(8,3))

    F(op2.direct(op1.direct(pdhg.solution)))
    F(op2.direct(op1.direct(VectorData(np.array([1,0])))))
    F(op2.direct(op1.direct(VectorData(np.array([1,-4.47])))))

    c = np.array([1,-4.47])
    show2D(op1.direct(VectorData(c)),title=f"c_ext={c}")

    c = np.array([1,-1])
    show2D(op1.direct(VectorData(c)),title=f"c_ext={c}")
    show1D(op1.direct(VectorData(c)), [('horizontal_x',hori_x)], title=f'horizontal_x={hori_x}', size=(8,3))

    def total_variation1(image):
        diff1 = np.diff(image, axis=0)
        diff2 = np.diff(image, axis=1)
        tv = np.sum(np.abs(diff1)) + np.sum(np.abs(diff2))
        return tv
    
    def total_variation2(image):
        dx = np.diff(image, axis=0)
        dy = np.diff(image, axis=1)
        dx = np.pad(dx, ((0, 1), (0, 0)), 'constant')
        dy = np.pad(dy, ((0, 0), (0, 1)), 'constant')
        tv = np.sum(np.sqrt(dx**2 + dy**2))
        return tv

    ##########
    plt.rcParams.update({'font.size': 16})
    vs = np.arange(-2.0,4.0,0.1)
    # vs = np.arange(-100,100,0.5)
    fs = np.zeros(vs.shape)
    for i,v in enumerate(vs):
        fs[i] = F(op2.direct(op1.direct(VectorData(np.array([1,v])))))
    
    fig,ax = plt.subplots(figsize=(8,5)); plt.sca(ax)
    plt.plot(vs,fs, color='black')
    plt.xlabel(r'$c_1$')
    plt.ylabel(r'Objective value')
    plt.plot(pdhg.solution.as_array()[1], pdhg.objective[-1], label='Minimizer',
        marker=".", markersize=15, color='red', linestyle='None')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig(os.path.join(base_dir, 'plots/shapes_obj_curve.pdf'))
    plt.show()
    plt.rcParams.update({'font.size': 12})

    ##########
    plt.rcParams.update({'font.size': 16})
    import matplotlib as mpl
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
    fig,ax = plt.subplots(1,2,figsize=(13,6))

    c = np.array([1,-1])
    Q_manual = op1.direct(VectorData(c))
    plt.sca(ax[0])
    plt.imshow(Q_manual.as_array(), origin='lower', cmap='gray')
    plt.colorbar()
    c_str = str(c[1:]).replace('[', '(').replace(']', ')')
    plt.title(rf'$\boldsymbol{{c}} = {c_str}$')

    Q = op1.direct(pdhg.solution)
    plt.sca(ax[1])
    plt.imshow(Q.as_array(), origin='lower', cmap='gray')
    plt.colorbar()
    rounded_c = [round(_, 3) for _ in pdhg.solution.as_array()[1:]]
    c_str = str(rounded_c).replace('[', '(').replace(']', ')')
    plt.title(rf'TV minimizer: $\boldsymbol{{c}} = {c_str}$')

    plt.tight_layout()
    # plt.savefig(os.path.join(base_dir, 'plots/shapes_ESC_sol_comparisons.pdf'))
    plt.show()
    #############

def esc_shapes_2(): # try with multiple bases
    from matplotlib import rc
    plt.rcParams.update({'font.size': 12})
    rc('text', usetex=True)
    rc('font', family='serif')
    
    data = load_centre('X20.pkl')
    ag = data.geometry
    ig = ag.get_ImageGeometry()

    filepath = os.path.join(base_dir,'test_images/test_image_shapes3.png')
    # filepath = os.path.join(base_dir,'test_images/esc_circles.png')
    # filepath = os.path.join(base_dir,'test_images/esc_geom.png')
    im_arr = io.imread(filepath)
    im_arr = color.rgb2gray(im_arr) > 0

    im = ImageData(array=im_arr.astype('float32'), geometry=ig)
    A = ProjectionOperator(ig, ag, 'Siddon', device='gpu')
    mu = 10 / (ig.voxel_num_x*ig.voxel_size_x) # scale to 1 mm
    # mu = 40 / (ig.voxel_num_x*ig.voxel_size_x)
    data = mu*A.direct(im)
    data.reorder('tigre')
    recon_P = FDK(data, image_geometry=ig).run(verbose=0)
    show2D(recon_P)

    ### Scatter simulation
    I_P = AbsorptionTransmissionConverter()(data).as_array()
    b = data.as_array()
    # factor = [5,20,40,100,200][2]
    # basis_params = [[0.15*1/factor,0,factor]]

    # factors = [40,100]
    factors = [60]
    basis_params = [[0.15*1/factor,0,factor] for factor in factors]

    I_S = compute_scatter_basis(b,basis_params)
    I_S_sum = np.sum(I_S,axis=0)/len(basis_params)
    # I_S = 0.05*I_S*3
    # I_S = 0.05*I_S*3
    I = I_P + I_S_sum

    basis_idx = 0
    idx = 0

    ########################
    fig,ax = plt.subplots(figsize=(8,5))
    plt.sca(ax)
    plt.plot(I_P[idx,:],label=r'$I_P$')
    # plt.plot(I_S[basis_idx,idx,:],label=r'$I_S$')
    plt.plot(I_S_sum[idx,:],label=r'$I_S$')
    # plt.plot(I[idx,:],label=r'I=I_P+I_S')
    plt.grid(True)
    plt.legend()
    plt.title('Projection with simulated scatter')
    # plt.savefig(os.path.join(base_dir, 'plots/shapes_scatter_proj.png'))
    plt.show()
    #####################

    data_scatter = TransmissionAbsorptionConverter()(AcquisitionData(array=np.array(I, dtype=np.float32), geometry=ag))
    recon = FDK(data_scatter, image_geometry=ig).run(verbose=0)

    ####################
    fig,ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(8,10))
    plt.sca(ax[0])
    im0 = plt.imshow(recon.as_array(), origin='lower', cmap='gray')#,vmin=-10, vmax=10)
    # plt.imshow(recon.as_array(), origin='lower', cmap='gray')
    plt.xlabel('horizontal_x')
    plt.ylabel('horizontal_y')
    plt.title(r'Reconstruction of $-\ln(I_P+I_S)$')
    plt.colorbar()
    # plt.colorbar(im0,fraction=0.046, pad=0.04)

    plt.sca(ax[1])
    hori_x,hori_y = 700,650

    plt.plot(recon.as_array()[:,hori_x]) # horizontal_x fixed
    plt.title(f'Intensity profile for horizontal_x={hori_x} on the reconstruction')
    plt.xlabel('horizontal_y')

    # plt.plot(recon.as_array()[hori_y,:]) # horizontal_y fixed
    # plt.title(f'Intensity profile for horizontal_y={hori_y} on the reconstruction')
    # plt.xlabel('horizontal_x')
    
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig(os.path.join(base_dir, 'plots/shapes_scatter_recon.pdf'))
    plt.show()
    ####################
    
    P = recon.as_array()
    b = data_scatter.as_array()
    ny,nx = ig.shape

    ### ESC step
    # basis_params = [[1/5,0,5],[1/10,0,10],[1/40,0,40]]

    # factor = [5,20,40,100,200][2]
    # basis_params = [[0.15*1/factor,0,factor]]
    # basis_params = [[0.15*1/factor,0,factor] for factor in factors]

    # factors = [40,100]
    factors = [60]
    basis_params = [[0.15*1/factor,0,factor] for factor in factors]

    trans = I
    nc = len(basis_params)

    basis_idx = 0
    if False:
        ### approximation (real world data)
        I_S = compute_scatter_basis(b,basis_params)
        I_S_sum = np.sum(I_S,axis=0)/len(basis_params)
        # I_S = 0.05*I_S
        # I_Q = trans-I_S
        I_Q = I - I_S_sum
        s = b[None,:,:] + np.log(I_Q)
        plt.plot(I_P[idx,:],label='I_P')
        plt.plot(I_Q[idx,:],label='I_Q')
        plt.plot(I[idx,:],label='I=I_P+I_S')
        plt.legend()
        plt.title(f'basis_idx: {basis_idx}')
        plt.grid(True)
        plt.show()
        ###
    else:
        ### what the model is based on
        I_S = compute_scatter_basis(-np.log(I_P),basis_params)
        I_S_sum = np.sum(I_S,axis=0)/len(basis_params)
        # I_S = 0.05*I_S
        # I_Q = trans-I_S
        I_Q = I - I_S_sum
        s = -np.log(I_S+I_P) + np.log(I_P)
        plt.plot(I_P[idx,:],label='I_P')
        plt.plot(I_Q[idx,:],label='I_Q')
        plt.plot(I[idx,:],label='I=I_P+I_S')
        plt.legend()
        plt.title(f'basis_idx: {basis_idx}')
        plt.grid(True)
        plt.show()
        ###

    # print(np.unravel_index(s.argmax(),s.shape), np.max(s))

    # fig,ax = plt.subplots(1,2,figsize=(10,5))
    # plt.title(f'basis_idx: {basis_idx}')
    # plt.sca(ax[0])
    # plt.plot(s[basis_idx,idx,:],label='s')
    # plt.legend()

    # plt.sca(ax[1])
    # plt.plot(-np.log(-s[basis_idx,idx,:]),label='-ln(-s)')
    # plt.legend()
    # plt.show()

    S = np.zeros((nc,*ig.shape))
    for i in range(nc):
        data_s_i = AcquisitionData(array=np.array(s[i], dtype='float32'), geometry=ag)
        S[i] = FDK(data_s_i).run(verbose=0).as_array()
        # show2D(-S[i])
        show2D(S[i],title=f'S_{i}')

    Mext = np.zeros((nx*ny,nc+1))
    Mext[:,0] = P.flatten()
    Mext[:,1:] = S.reshape(nc, nx*ny).T

    vg = VectorGeometry(length=nc+1)
    op1 = ESCMatrixOperator(Mext, domain_geometry=vg, range_geometry=ig)
    op2 = GradientOperator(domain_geometry=ig)
    K = CompositionOperator(op2,op1)
    F = MixedL21Norm()
    # F = MixedL11Norm()
    # F = SmoothMixedL21Norm(epsilon=1e-2)

    cext = VectorData(array=np.hstack((1,np.full(nc, 0.1, dtype=np.float32))), geometry=vg)

    l = -np.inf*np.ones(1+nc)
    l[0] = 1
    u = np.inf*np.ones(1+nc)
    u[0] = 1
    G = IndicatorBox(lower=l,upper=u)
    upd_int = 3
    pdhg = PDHG(f=F, g=G, operator=K, initial=cext, max_iteration=1000, update_objective_interval=3)
    # pdhg.run(iterations=150)
    pdhg.run(iterations=34)

    print(pdhg.solution.as_array())
    Q = op1.direct(pdhg.solution)
    show2D(Q, title=f"c_ext={pdhg.solution.as_array()}")
    show1D(Q, [('horizontal_x',hori_x)], title=f'horizontal_x={hori_x}', size=(8,3))
    # show1D(Q, [('horizontal_y',hori_y)], title=f'horizontal_y={hori_y}', size=(8,3))

    F(op2.direct(op1.direct(pdhg.solution)))
    F(op2.direct(op1.direct(VectorData(np.array([1,0])))))
    F(op2.direct(op1.direct(VectorData(np.array([1,-4.47])))))

    c = np.array([1,-4.47])
    show2D(op1.direct(VectorData(c)),title=f"c_ext={c}")

    c = np.array([1,-1])
    show2D(op1.direct(VectorData(c)),title=f"c_ext={c}")
    show1D(op1.direct(VectorData(c)), [('horizontal_x',hori_x)], title=f'horizontal_x={hori_x}', size=(8,3))

    def total_variation1(image):
        diff1 = np.diff(image, axis=0)
        diff2 = np.diff(image, axis=1)
        tv = np.sum(np.abs(diff1)) + np.sum(np.abs(diff2))
        return tv
    
    def total_variation2(image):
        dx = np.diff(image, axis=0)
        dy = np.diff(image, axis=1)
        dx = np.pad(dx, ((0, 1), (0, 0)), 'constant')
        dy = np.pad(dy, ((0, 0), (0, 1)), 'constant')
        tv = np.sum(np.sqrt(dx**2 + dy**2))
        return tv

    ##########
    plt.rcParams.update({'font.size': 16})
    vs = np.arange(-2.0,4.0,0.1)
    # vs = np.arange(-100,100,0.5)
    fs = np.zeros(vs.shape)
    for i,v in enumerate(vs):
        fs[i] = F(op2.direct(op1.direct(VectorData(np.array([1,v])))))
    
    fig,ax = plt.subplots(figsize=(8,5)); plt.sca(ax)
    plt.plot(vs,fs, color='black')
    plt.xlabel(r'$c_1$')
    plt.ylabel(r'Objective value')
    plt.plot(pdhg.solution.as_array()[1], pdhg.objective[-1], label='Minimizer',
        marker=".", markersize=15, color='red', linestyle='None')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig(os.path.join(base_dir, 'plots/shapes_obj_curve.pdf'))
    plt.show()
    plt.rcParams.update({'font.size': 12})

    ##########
    plt.rcParams.update({'font.size': 16})
    import matplotlib as mpl
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
    fig,ax = plt.subplots(1,2,figsize=(13,6))

    c = np.array([1,-1])
    Q_manual = op1.direct(VectorData(c))
    plt.sca(ax[0])
    plt.imshow(Q_manual.as_array(), origin='lower', cmap='gray')
    plt.colorbar()
    c_str = str(c[1:]).replace('[', '(').replace(']', ')')
    plt.title(rf'$\boldsymbol{{c}} = {c_str}$')

    Q = op1.direct(pdhg.solution)
    plt.sca(ax[1])
    plt.imshow(Q.as_array(), origin='lower', cmap='gray')
    plt.colorbar()
    rounded_c = [round(_, 3) for _ in pdhg.solution.as_array()[1:]]
    c_str = str(rounded_c).replace('[', '(').replace(']', ')')
    plt.title(rf'TV minimizer: $\boldsymbol{{c}} = {c_str}$')

    plt.tight_layout()
    # plt.savefig(os.path.join(base_dir, 'plots/shapes_ESC_sol_comparisons.pdf'))
    plt.show()
    #############

    fig,ax = plt.subplots(1,1,figsize=(8,7))

    plt.sca(ax)
    plt.imshow(S[0], origin='lower', cmap='gray')
    plt.colorbar()
    plt.title(rf'$\boldsymbol{{S}}_1$')

    plt.tight_layout()
    # plt.savefig(os.path.join(base_dir, 'plots/shapes_ESC_basis.pdf'))
    plt.show()
    ###############

def esc_X20():
    from matplotlib import rc
    plt.rcParams.update({'font.size': 12})
    rc('text', usetex=True)
    rc('font', family='serif')

    data = load_centre('X20_cor.pkl')
    ag = data.geometry
    ig = ag.get_ImageGeometry()
    A = ProjectionOperator(ig, ag, direct_method='Siddon', device='gpu')
    recon = FDK(data, image_geometry=ig).run(verbose=0)

    I = AbsorptionTransmissionConverter()(data).as_array()
    b = data.as_array()
    ny,nx = ig.shape
    P = recon.as_array()

    ### ESC step
    # basis_params = [[1/5,0,5],[1/10,0,10],[1/40,0,40]]

    # factor = [5,20,40,100,200][2]
    # basis_params = [[0.15*1/factor,0,factor]]
    # basis_params = [[0.15*1/factor,0,factor] for factor in factors]

    # factors = [40,100]
    factors = [40]
    basis_params = [[0.05*1/factor,0,factor] for factor in factors]

    trans = I
    nc = len(basis_params)

    idx = 800
    basis_idx = 0
    ### approximation (real world data)
    I_S = compute_scatter_basis(b,basis_params)
    I_S_sum = np.sum(I_S,axis=0)/len(basis_params)
    # I_S = 0.05*I_S
    # I_Q = trans-I_S
    I_Q = I - I_S_sum
    s = b[None,:,:] + np.log(I_Q)
    plt.plot(I_S_sum[idx,:],label='I_S')
    plt.plot(I[idx,:],label='I')
    plt.legend()
    plt.title(f'basis_idx: {basis_idx}')
    plt.grid(True)
    plt.show()

    idx = 1120
    fig,ax = plt.subplots(figsize=(8,5))
    plt.sca(ax)
    plt.plot(I[idx,:],label=r'$I$')
    # plt.plot(I_S[basis_idx,idx,:],label=r'$I_S$')
    plt.plot(I_S_sum[idx,:],label=r'$I_S$')
    # plt.plot(I[idx,:],label=r'I=I_P+I_S')
    plt.grid(True)
    plt.legend()
    plt.title('Projection from X20 with scatter estimation')
    # plt.savefig(os.path.join(base_dir, 'plots/X20_scatter_proj.png'))
    plt.show()

    print(np.unravel_index(s.argmax(),s.shape), np.max(s))

    # fig,ax = plt.subplots(1,2,figsize=(10,5))
    # plt.title(f'basis_idx: {basis_idx}')
    # plt.sca(ax[0])
    # plt.plot(s[basis_idx,idx,:],label='s')
    # plt.legend()

    # plt.sca(ax[1])
    # plt.plot(-np.log(-s[basis_idx,idx,:]),label='-ln(-s)')
    # plt.legend()
    # plt.show()

    S = np.zeros((nc,*ig.shape))
    for i in range(nc):
        data_s_i = AcquisitionData(array=np.array(s[i], dtype='float32'), geometry=ag)
        S[i] = FDK(data_s_i).run(verbose=0).as_array()
        # show2D(-S[i])
        show2D(S[i],title=f'S_{i}')

    Mext = np.zeros((nx*ny,nc+1))
    Mext[:,0] = P.flatten()
    Mext[:,1:] = S.reshape(nc, nx*ny).T

    vg = VectorGeometry(length=nc+1)
    op1 = ESCMatrixOperator(Mext, domain_geometry=vg, range_geometry=ig)
    op2 = GradientOperator(domain_geometry=ig)
    K = CompositionOperator(op2,op1)
    F = MixedL21Norm()
    # F = MixedL11Norm()
    # F = SmoothMixedL21Norm(epsilon=1e-2)

    cext = VectorData(array=np.hstack((1,np.full(nc, 0.1, dtype=np.float32))), geometry=vg)

    l = -np.inf*np.ones(1+nc)
    l[0] = 1
    u = np.inf*np.ones(1+nc)
    u[0] = 1
    G = IndicatorBox(lower=l,upper=u)
    upd_int = 3
    pdhg = PDHG(f=F, g=G, operator=K, initial=cext, max_iteration=1000, update_objective_interval=3)
    # pdhg.run(iterations=150)
    # pdhg.run(iterations=34)
    pdhg.run(iterations=30)

    print(pdhg.solution.as_array())
    Q = op1.direct(pdhg.solution)
    show2D(Q, title=f"c_ext={pdhg.solution.as_array()}")

    F(op2.direct(op1.direct(pdhg.solution)))
    F(op2.direct(op1.direct(VectorData(np.array([1,0])))))
    F(op2.direct(op1.direct(VectorData(np.array([1,-4.47])))))

    c = np.array([1,-10*0])
    show2D(op1.direct(VectorData(c)),title=f"c_ext={c}")

    c = np.array([1,-1])
    show2D(op1.direct(VectorData(c)),title=f"c_ext={c}")

    ##########
    plt.rcParams.update({'font.size': 16})
    vs = np.arange(-2.0,4.0,0.1)
    # vs = np.arange(-100,100,0.5)
    fs = np.zeros(vs.shape)
    for i,v in enumerate(vs):
        fs[i] = F(op2.direct(op1.direct(VectorData(np.array([1,v])))))
    
    fig,ax = plt.subplots(figsize=(8,5)); plt.sca(ax)
    plt.plot(vs,fs, color='black')
    plt.xlabel(r'$c_1$')
    plt.ylabel(r'Objective value')
    plt.plot(pdhg.solution.as_array()[1], pdhg.objective[-1], label='Minimizer',
        marker=".", markersize=15, color='red', linestyle='None')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig(os.path.join(base_dir, 'plots/shapes_obj_curve.pdf'))
    plt.show()
    plt.rcParams.update({'font.size': 12})
    
    ##########
    imshow_kwargs = {
        'origin': 'lower', 'cmap': 'gray'
        # ,'vmin': -0.4, 'vmax': 0.8
        ,'vmin': -0.6, 'vmax': 1.2
    }
    
    slice_y = slice(350,700)
    plt.rcParams.update({'font.size': 16})
    import matplotlib as mpl
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
    fig,ax = plt.subplots(3,1,figsize=(9,12))

    c = np.array([1,-3])
    Q_manual = op1.direct(VectorData(c))
    plt.sca(ax[0])
    plt.imshow(Q_manual.as_array()[slice_y], **imshow_kwargs)
    plt.colorbar()
    c_str = str(c[1:]).replace('[', '(').replace(']', ')')
    plt.title(rf'$\boldsymbol{{c}} = {c_str}$')

    c = np.array([1,0])
    Q_manual = op1.direct(VectorData(c))
    plt.sca(ax[1])
    plt.imshow(Q_manual.as_array()[slice_y], **imshow_kwargs)
    plt.colorbar()
    c_str = str(c[1:]).replace('[', '(').replace(']', ')')
    plt.title(rf'$\boldsymbol{{c}} = {c_str}$')

    Q = op1.direct(pdhg.solution)
    plt.sca(ax[2])
    plt.imshow(Q.as_array()[slice_y], **imshow_kwargs)
    plt.colorbar()
    rounded_c = [round(_, 3) for _ in pdhg.solution.as_array()[1:]]
    c_str = str(rounded_c).replace('[', '(').replace(']', ')')
    plt.title(rf'TV minimizer: $\boldsymbol{{c}} = {c_str}$')

    plt.tight_layout()
    # plt.savefig(os.path.join(base_dir, 'plots/X20_ESC_sol_comparisons.pdf'))
    plt.show()
    #############

    show2D(-S[0])
    S_seg = clip_otsu_segment(-S[0], ig, clip=0)
    show2D(S_seg)

    imshow_kwargs = {
        'origin': 'lower',
        'cmap': 'gray'
        # ,'vmin': -0.4,
        # 'vmax': 0.8
    }
    fig,ax = plt.subplots(3,1,figsize=(9,12))

    plt.sca(ax[0])
    plt.imshow(S[0][slice_y], origin='lower', cmap='gray')
    plt.colorbar()
    plt.title(rf'$\boldsymbol{{S}}_1$')

    plt.sca(ax[1])
    plt.imshow(-S[0][slice_y], **imshow_kwargs)
    plt.colorbar()
    plt.title(rf'$-\boldsymbol{{S}}_1$')

    plt.sca(ax[2])
    plt.imshow(recon.as_array()[slice_y], **imshow_kwargs)
    plt.colorbar()
    plt.title(rf'$\boldsymbol{{Q}}$')

    plt.tight_layout()
    # plt.savefig(os.path.join(base_dir, 'plots/X20_ESC_basis.pdf'))
    plt.show()

    ###########
    c = np.array([1,-5])
    recon_cor = op1.direct(VectorData(c))
    show2D(recon_cor,title=f"c_ext={c}")

    data_cor = b + np.sum(c[:,None,None]*I_S, axis=0)
    data_cor = AcquisitionData(array=np.array(data_cor, dtype='float32'), geometry=ag)

    segmentation = clip_otsu_segment(recon_cor.as_array(), ig, clip=0)
    path_lengths = A.direct(segmentation)

    mask = (path_lengths.as_array() > 0.05) & (data.as_array() > 0.25)
    def f_poly1(x, *a):
        return a[0]*x**3
    def f_poly2(x, *a):
        return a[0]*x**5
    
    shift = 0.05
    const = np.log10(shift)
    def f_poly3(x, *a):
        # return a[0]*np.log10(x+0.1) + 1
        # return 10**((x-1)/a[0])-0.1
        return 10**((x+const)/a[0]) - shift

    bhc = BHC(path_lengths, data, None, f_poly1, num_bins=100, mask=mask, n_poly=1)
    bhc.run()
    ########

def disk_sino():
    physical_size = 1
    voxel_num = 1000
    ag,ig = setup_generic_cil_geometry(physical_size=1,voxel_num=voxel_num)
    # im_arr = create_circle_image(image_size=voxel_num, radius=voxel_num//5, center=[voxel_num//2, voxel_num//2])
    im_arr = create_circle_image(image_size=voxel_num, radius=voxel_num//6, center=[voxel_num//4, voxel_num//4])
    im = ImageData(array=im_arr.astype('float32'), geometry=ig)
    
    A = ProjectionOperator(ig, ag, direct_method='Siddon', device='gpu')
    data = A.direct(im)
    recon = FBP(data, image_geometry=ig).run(verbose=0)
    show2D(recon)
    show2D(data)