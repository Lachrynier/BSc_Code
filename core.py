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



def fun_attenuation(plot=False):
    """
    Reads in NIST data and converts it to proper format
    Plots the attenuation coefficients against energy if plot is True
    Returns an attenuation function that is a linear interpolation of data in log-log domain
    """

    # mu/rho in (cm2/g)
    # multiply by density of gold to get mu in (cm^{-1})
    # original data in MeV
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
    # print(f'rho: {rho_gold}')
    # divide by 10 to get it in mm
    mu = data[:,1] * rho_gold / 10 # mm^{-1}

    # Perform spline interpolation in log domain. k=1 is linear
    spline_log_domain = interpolate.InterpolatedUnivariateSpline(np.log10(energies), np.log10(mu), k=1)
    def estimate_attenuation(energy):
        return np.power(10, spline_log_domain(np.log10(energy)))
    
    energy_plot_range = np.logspace(np.log10(min(energies)), np.log10(max(energies)), 500)

    if plot == True:
        plt.loglog(energy_plot_range, estimate_attenuation(energy_plot_range), label='Spline Interpolation (Linear)')
        plt.loglog(energies, mu, 'o', label='Data Points',markersize=4)#,color='black')  # Data points

        plt.xlabel('Energy (keV)')
        plt.ylabel('Attenuation coefficient (mm^{-1})')
        plt.title('Attenuation coefficient vs energy')
        plt.legend()
        plt.show()

    return estimate_attenuation

def generate_spectrum(plot=False,filter=0,tube_potential=220,bin_width=2):
    """
    Generates normalized and binned X-ray spectrum using SpekPy.
    Plots the spectrum if plot is True

    tube_potential in kV
    bin_width in keV
    filter in mm
    
    Returns lists of the bin centers and heights
    """

    # anode angle in degrees
    # anode_angle = 90
    anode_angle = 12

    # data_source = 'pene'
    data_source = 'nist'

    physics_arr = ['diff','spekpy-v1','casim']
    physics = physics_arr[2]
    spectrum = spekpy.Spek(kvp=tube_potential, th=anode_angle, dk=bin_width, mu_data_source=data_source, physics=physics)
    spectrum.filter('Sn',filter)
    spectrum = np.array(spectrum.get_spectrum())
    spectrum[1,:] /= np.sum(spectrum[1,:])

    num_bins = spectrum.shape[1]

    bin_centers = spectrum[0,:]
    bin_heights = spectrum[1,:]
    bin_edges = np.array(bin_centers) - bin_width/2
    bin_edges = np.append(bin_edges, bin_edges[-1] + bin_width)

    if plot == True:
        plt.bar(bin_centers, bin_heights, align='center', width=np.diff(bin_edges))
        plt.title(f'Normalized histogram of X-ray spectrum at {tube_potential} keV')
        plt.show()
    
    return bin_centers, bin_heights

def create_circle_image(image_size, radius, center):
    image = np.zeros((image_size, image_size))
    for x in range(image_size):
        for y in range(image_size):
            if (x - center[0]) ** 2 + (y - center[1]) ** 2 < (radius+0.5) ** 2:
                image[x, y] = 1
    
    return image

def generate_triangle_image(img_size, triangle_size, corner_coords):
    """
    Generate a right triangle test image.

    Parameters:
    img_size (tuple): Size of the image (width, height)
    triangle_size (tuple): Size of the right triangle (base, height)
    corner_coords (tuple): Coordinates of the right-angled corner (x, y)
    """
    image = np.zeros(img_size)

    # Define the triangle
    for x in range(corner_coords[0], min(corner_coords[0] + triangle_size[0], img_size[0])):
        for y in range(corner_coords[1], min(corner_coords[1] + triangle_size[1], img_size[1])):
            if (x - corner_coords[0]) + (y - corner_coords[1]) < triangle_size[0]:
                image[y, x] = 1

    return image

def create_rings_image(size, ring_width, spacing, radius, center):
    """
    Create a square image with concentric rings.

    Parameters:
    - size: The size of the square image (width and height) in pixels.
    - ring_width: The width of each ring in pixels.
    - spacing: The spacing between rings in pixels.
    - radius: The overall radius of the circles (from the center to the outer edge) in pixels.
    - center: A tuple (x, y) representing the center of the circles.
    """

    image = np.zeros((size, size))

    # Generate a grid of distances from the center
    Y, X = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

    # Create rings by selecting pixels within specific distance ranges
    for r in np.arange(radius, 0, -(ring_width + spacing)):
        inner_radius = r - ring_width
        mask = (dist_from_center <= r) & (dist_from_center > inner_radius)
        image[mask] = 1

    return image

def load_centre(filename):
    """
    Assumes that centres are stored at base_dir/centres/ as 
    pickle objects containing CIL AcquisitionsData
    """

    # base_dir = os.path.abspath('/dtu/3d-imaging-center/projects/2022_DANFIX_Vindelev/analysis/s214743_bsc/')
    file_path = os.path.join(base_dir,f'centres/{filename}')
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    return data

def bhc_alt(path_lengths, data, f_mono, f_poly, num_bins):
    """
    Alternative linearization BHC based on unweighted binning.
    Returns the BHC data and FDK reconstruction of this.
    Parameters the same subset as that of the main BHC class.
    """
    def avg_out(path_lengths, data, num_bins):
        x = np.array(path_lengths.as_array().flatten())
        y = np.array(data.as_array().flatten())
        bin_edges = np.linspace(x.min(), x.max(), num_bins+1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_indices = np.digitize(x, bin_edges)
        bin_means = np.array([y[bin_indices == i].mean() for i in range(1, len(bin_edges))])
        plt.scatter(x, y, alpha=0.1, color='black', label='Observed',s=2)
        # plt.plot(bin_centers, bin_means, color='red', label='Averaged out')
        plt.plot(bin_centers, bin_means, color='red', marker='o', linestyle='-', label='Averaged out')
        plt.xlabel('Path lengths (mm)')
        plt.ylabel('Absorption')
        plt.legend(loc='lower right')
        return np.hstack(([0],bin_centers)), np.hstack(([0],bin_means))

    bin_centers,bin_means = avg_out(path_lengths, data, num_bins=num_bins)
    popt_mono, pcov = curve_fit(f_mono, bin_centers, bin_means)
    popt_poly, pcov = curve_fit(f_poly, bin_means, bin_centers)
    xx = np.linspace(0, path_lengths.max())
    yy = np.linspace(0, data.max())
    y_mono = f_mono(xx, *popt_mono)
    x_poly = f_poly(yy, *popt_poly)
    plt.plot(bin_centers, bin_means, color='red', marker='o', linestyle='-', label='Averaged out')
    plt.plot(xx, y_mono, label='Mono fit')
    plt.plot(x_poly, yy, label='Poly fit')
    plt.legend(loc='lower right')
    plt.show()

    # gglin = g(yylin,popt_g[0],popt_g[1])
    yy_bhc = f_mono(x_poly, *popt_mono)
    plt.plot(yy, yy_bhc)
    plt.title('BHC absorptions vs Original absorptions')
    plt.show()

    data_x_poly = f_poly(data.as_array(), *popt_poly)
    data_y_mono = f_mono(data_x_poly, *popt_mono)
    data_bhc = AcquisitionData(array=np.array(data_y_mono,dtype='float32'),geometry=data.geometry)
    recon_bhc = FDK(data_bhc).run()
    return data_bhc, recon_bhc

class BHC:
    """
    Class representation of the BHC method.
    Call the run method to to apply BHC after initialization.

    path_lengths: AcquisitionData object of path lengths
    data: AcquisitionData object of data
    f_mono: function handle for the monochromatic fit. Can be omitted, which will default to a line.
    f_poly:
        function handle for the polychromatic fit. 
        Should have the inputs (x, *a) where x is vectorized evaluation and
        a are the coefficients.
    n_poly:
        the number of parameters of f_poly. this does not have to be explicitly stated if
        the coefficients a of f_poly are not unpacked, such that you have to pass a0,a1,a2,...
    num_bins: number of bins for the 2D histogram
    mask: binary mask on data indicating which to use for the 2D histogram and fits.
    weight_fun: see the documentation for the variable sigma of scipy.optimize.curve_fit
    color_norm: plot argument for the 2D histogram
    """
    def __init__(self, path_lengths, data, f_mono, f_poly, num_bins, mask=None, weight_fun=np.sqrt, color_norm='log', n_poly=None):
        self.path_lengths = path_lengths
        self.data = data
        self.f_mono = f_mono
        self.f_poly = f_poly
        self.num_bins = num_bins
        self.mask = mask
        self.weight_fun = weight_fun
        self.color_norm = color_norm
        self.n_poly = n_poly
        self.popt_poly = None
        self.popt_mono = None
        self.data_bhc = None
        self.recon_bhc = None
        self.x_centers_flat = None
        self.y_centers_flat = None
        self.counts_flat = None

        if f_mono is None:
            def f_mono(x, a):
                return a*x
            self.f_mono = f_mono

    def prepare_data(self):
        if self.mask is None:
            x = np.array(self.path_lengths.as_array()).flatten()
            y = np.array(self.data.as_array()).flatten()
        else:
            x = np.array(self.path_lengths.as_array()[self.mask])
            y = np.array(self.data.as_array()[self.mask])
        return x, y

    def plot_histogram(self, x, y):
        norms = {'lin': Normalize(), 'log': LogNorm()}
        counts, x_edges, y_edges, _ = plt.hist2d(x, y, bins=self.num_bins, range=[[0, x.max()], [0, y.max()]], norm=norms[self.color_norm])
        plt.xlabel('Path length [mm]')
        plt.ylabel('Absorption')
        plt.colorbar()
        return counts, x_edges, y_edges

    def calculate_fit_data(self, counts, x_edges, y_edges):
        x_centers = 0.5 * (x_edges[1:] + x_edges[:-1])
        y_centers = 0.5 * (y_edges[1:] + y_edges[:-1])
        y_centers, x_centers = np.meshgrid(y_centers, x_centers)
        self.x_centers_flat = x_centers.flatten()
        self.y_centers_flat = y_centers.flatten()
        self.counts_flat = counts.flatten()

    def perform_fit(self):
        nonzero_counts = self.counts_flat > 0
        x_fit = self.x_centers_flat[nonzero_counts]
        y_fit = self.y_centers_flat[nonzero_counts]
        counts_fit = self.counts_flat[nonzero_counts]
        weights = self.weight_fun(counts_fit)
        
        self.popt_mono, _ = curve_fit(self.f_mono, x_fit, y_fit, sigma=1./weights, absolute_sigma=True)
        if self.n_poly is not None:
            p0 = np.ones(self.n_poly)
        else:
            p0 = None
        self.popt_poly, _ = curve_fit(self.f_poly, y_fit, x_fit, p0=p0, sigma=1./weights, absolute_sigma=True)

    def plot_fits(self, show_hist=True, make_trans_plot=True, **kwargs):
        xx = np.linspace(0, self.data.max() / self.popt_mono)
        yy = np.linspace(0, self.data.max())
        y_mono = self.f_mono(xx, *self.popt_mono)
        x_poly = self.f_poly(yy, *self.popt_poly)
        lw = 3
        if not kwargs:
            plt.plot(x_poly, yy, label='Poly fit', color='red', linewidth=lw)
            plt.title('Fit used for linearization')
        else:
            plt.plot(x_poly, yy, **kwargs)
        
        if show_hist:
            plt.show()
        if make_trans_plot:
            plt.plot(yy, self.f_mono(x_poly, *self.popt_mono))
            plt.title('BHC absorptions vs Original absorptions')
            plt.grid(True)
            plt.show()

    def perform_correction(self):
        data_x_poly = self.f_poly(self.data.as_array(), *self.popt_poly)
        data_y_mono = self.f_mono(data_x_poly, *self.popt_mono)
        self.data_bhc = AcquisitionData(array=np.array(data_y_mono, dtype='float32'), geometry=self.data.geometry)
        self.recon_bhc = FDK(self.data_bhc).run(verbose=0)

    def run(self, verbose=1):
        x, y = self.prepare_data()
        counts, x_edges, y_edges = self.plot_histogram(x, y)
        self.calculate_fit_data(counts, x_edges, y_edges)
        self.perform_fit()
        if verbose == 1:
            self.plot_fits()
        else:
            self.plot_fits(show_hist=False,make_trans_plot=False)
            plt.close()
        self.perform_correction()
        return self.data_bhc, self.recon_bhc
    
    def get_hist_fit_plot(self):
        ### Auxiliary function for extended work on the hist plot

        x, y = self.prepare_data()
        counts, x_edges, y_edges = self.plot_histogram(x, y)
        self.calculate_fit_data(counts, x_edges, y_edges)
        self.perform_fit()
        # self.plot_fits(show_hist=False, make_trans_plot=False)

def lin_interp_sino2D(data,tau):
    """
    Peforms linear interpolation on a projection-basis based on a threshold.

    data: AcquisitionData object
    tau: scalar threshold
    """
    def lin_interp_proj(proj):
        # Linear interpolation of a single projections
        above_tau_indices = np.where(proj > tau)[0]
        diffs = np.diff(above_tau_indices)
        breaks = np.where(diffs > 1)[0]
        groups = np.split(above_tau_indices, breaks+1)
        if not list(groups[0]):
            return proj
        for group in groups:
            if group[0] == 0:
                lval = tau
            else:
                lval = proj[group[0]-1]
            if group[-1] == proj.size-1:
                rval = tau
            else:
                rval = proj[group[-1]+1]
            
            proj[group] = (rval-lval)/((group[-1]+1) - (group[0]-1)) * (group - (group[0]-1)) + lval
        return proj
    
    # data_interp = np.apply_along_axis(func1d=lin_interp_proj, axis=1, arr=data.copy().as_array())
    data_interp = data.copy().as_array()
    for i in range(data.shape[0]):
        data_interp[i,:] = lin_interp_proj(data_interp[i,:])
    return AcquisitionData(array=data_interp, geometry=data.geometry)

def make_projection_plot(
        data,image,angle_idx,horizontal_idxs,
        show_source=False,scale_proj_axis=1.2,scale_factor=1.8,
        ray_opacity=0.5,ray_thickness=1.5,scale_abs=0.3,
        random_colors=True,max_type='absorption',axis_color='black',
        bound_color='red', scale_factor_x=None,scale_factor_y=None,lims=None,
        show=True
    ):
    """
    Embed projections on top of an image such as a reconstruction image.
    Currently assumes orthogonality of center ray with detector panel. That is, detector_direction_x = [1,0].

    data: AcquisitionData object
    image: Image that data is embedded on top of
    angle_idx: angle index of data
    horizontal_idxs: list of ray indices to include
    The rest of the parameters control the plotting details
    """
    def R_to_plot(vec):
        return axis_scalings*(vec + shift_R_to_I)
    
    ag = data.geometry
    ig = ag.get_ImageGeometry()
    angle = ag.angles[angle_idx]

    ### initializations
    voxel_num_x, voxel_num_y, voxel_num_z = ig.voxel_num_x, ig.voxel_num_y, ig.voxel_num_z
    voxel_size_x, voxel_size_y, voxel_size_z = ig.voxel_size_x, ig.voxel_size_y, ig.voxel_size_z
    center_x, center_y, center_z = ig.center_x, ig.center_y, ig.center_z
    image_max_side = np.max([voxel_num_x*voxel_size_x, voxel_num_y*voxel_size_y])

    axis_scalings = 1/np.array([voxel_size_x,voxel_size_y])
    shift_image_origin = 1/axis_scalings * np.array([voxel_num_x,voxel_num_y]) / 2

    source_position = ag.config.system.source.position
    rotation_axis_position = ag.config.system.rotation_axis.position
    detector_position = ag.config.system.detector.position
    detector_direction_x = ag.config.system.detector.direction_x

    shift_rot_axis_origin = -rotation_axis_position
    angle *= np.pi/180
    rot_mat_angle = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])

    shift_R_to_I = -shift_rot_axis_origin + shift_image_origin

    panel_num_pixels = ag.config.panel.num_pixels
    panel_pixel_size = ag.config.panel.pixel_size
    panel_origin = ag.config.panel.origin

    rot_source_R = rot_mat_angle.T @ (source_position + shift_rot_axis_origin)
    # rot_source_R = rot_mat_angle.T @ (source_position - rotation_axis_position)
    rot_source = rot_source_R - shift_rot_axis_origin
    rot_source_I = rot_source_R + shift_image_origin

    ### line towards detector position
    rot_panel_center_R = rot_mat_angle.T @ (detector_position + shift_rot_axis_origin)
    rot_panel_center = rot_panel_center_R - shift_rot_axis_origin
    rot_panel_center_I = rot_panel_center_R + shift_image_origin

    # theta,rho = calc_line_params_from_two_points(rot_source_R,rot_panel_pixel_R)
    # rot_mat_theta = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    n = (rot_panel_center_R - rot_source_R) / np.linalg.norm(rot_panel_center_R - rot_source_R,2)
    n_3hat = -np.array([n[1],-n[0]])
    scale_n = np.linalg.norm(source_position, 2) + scale_proj_axis * image_max_side/2
    L = np.linalg.norm(detector_position-source_position,2)
    
    ###
    def horizontal_to_proj(horizontal,scale,proj_abs=None):
        panel_pixel_position = detector_position + detector_direction_x * panel_pixel_size * ( -(panel_num_pixels[0]-1)/2 + horizontal )
        rot_panel_pixel_R = rot_mat_angle.T @ (panel_pixel_position + shift_rot_axis_origin)
        rot_panel_pixel = rot_panel_pixel_R - shift_rot_axis_origin
        rot_panel_pixel_I = rot_panel_pixel_R + shift_image_origin

        H = np.linalg.norm(panel_pixel_position - detector_position, 2)
        scale_n_3hat = H/L * scale_n
        if horizontal < (panel_num_pixels[0] // 2):
            scale_n_3hat *= (-1)
    
        # proj_x is "x-axis" of fan-beam projection plot
        proj_x_R = rot_source_R + scale_n*n + scale_n_3hat*n_3hat
        if scale == None:
            proj_y_R = None
        else:
            if not proj_abs:
                proj_abs = data.as_array()[angle_idx,horizontal]
            # proj_y_R = proj_x_R + 0.1*image_max_side * n
            proj_y_R = proj_x_R + scale*image_max_side * proj_abs/max_abs * n

        # print(f'Horizontal {horizontal}: {proj_x_R,proj_y_R}')
        return proj_x_R,proj_y_R

    if show:
        plt.figure(figsize=[10,10])
        # plt.figure(figsize=[20,20])
    # plt.imshow(ig.allocate(0.0).as_array(),cmap='gray')
    # plt.imshow(image.as_array(),cmap='gray',vmin=image.min(),vmax=image.max())
    plt.imshow(image.as_array(),origin='lower',cmap='gray',vmin=image.min(),vmax=image.max())
    xlims = plt.xlim()
    ylims = plt.ylim()

    # plt.plot(*zip(R_to_plot(rot_panel_center_R - panel_num_pixels[0]//2*n_3hat),R_to_plot(rot_panel_center_R + panel_num_pixels//2*n_3hat)),'k-')#,linewidth=5)

    # max_abs = data.as_array().max()
    max_abs = np.max(data.as_array()[angle_idx,horizontal_idxs])
    print(f'Max {max_type} for projection: {np.max(data.as_array()[angle_idx,:])}')
    plot_xs = []
    plot_ys = []
    color_abs = 'magenta'
    color_list = list(matplotlib.colors.cnames)
    color_list = [_ for _ in color_list if _ not in ['cyan',axis_color,bound_color]]

    # color_cycle = itertools.cycle(color_list)
    for horizontal in horizontal_idxs:
        proj_x_R,proj_y_R = horizontal_to_proj(horizontal,scale=scale_abs)

        if random_colors:
            color_ray = random.choice(color_list)
            # color_ray = next(color_cycle)
            color_abs = color_ray
            plt.plot(*zip(R_to_plot(rot_source_R),R_to_plot(proj_x_R)),'-',linewidth=ray_thickness, alpha=ray_opacity,color=color_ray)
            plt.plot(*zip(R_to_plot(proj_x_R),R_to_plot(proj_y_R)),'-',linewidth=ray_thickness,color=color_abs)
        else:
            plt.plot(*zip(R_to_plot(rot_source_R),R_to_plot(proj_x_R)),'r-',linewidth=ray_thickness, alpha=ray_opacity)
            plt.plot(*zip(R_to_plot(proj_x_R),R_to_plot(proj_y_R)),'b-',linewidth=ray_thickness,color=color_abs)

        # print(f'Magenta: {R_to_plot(proj_x_R),R_to_plot(proj_y_R)}')
        plot_xs.append(R_to_plot(proj_y_R)[0])
        plot_ys.append(R_to_plot(proj_y_R)[1])
        # plt.legend(loc='upper left')

    # plt.plot(plot_xs,plot_ys,'-o',color='cyan',markersize=2,label=f'Maximum visible absorption: {max_abs:.3f}')
    plt.plot(plot_xs,plot_ys,'-o',color='cyan',markersize=2,label=f'Absorption values')

    left_end,_ = horizontal_to_proj(0-0.5,scale=None)
    right_end,_ = horizontal_to_proj(panel_num_pixels[0]+0.5,scale=None)
    plt.plot(*zip(R_to_plot(left_end),R_to_plot(right_end)),'--',color=axis_color,label=f'Absorption = {0*max_abs:.2f}')

    ### upper bar
    max_pad = 1.1
    _,top_left_end = horizontal_to_proj(0-0.5,scale=scale_abs,proj_abs=1.1*max_abs)
    _,top_right_end = horizontal_to_proj(panel_num_pixels[0]+0.5,scale=scale_abs,proj_abs=1.1*max_abs)
    plt.plot(*zip(R_to_plot(top_left_end),R_to_plot(top_right_end)),'--',color=bound_color,label=f'Absorption = {max_pad*max_abs:.2f}')
    ###

    # Calculate new limits
    if not scale_factor_x: scale_factor_x = scale_factor
    if not scale_factor_y: scale_factor_y = scale_factor
    new_xlims = ((xlims[0] + xlims[1]) / 2 - (xlims[1] - xlims[0]) / 2 * scale_factor_x, 
                (xlims[0] + xlims[1]) / 2 + (xlims[1] - xlims[0]) / 2 * scale_factor_x)

    new_ylims = ((ylims[0] + ylims[1]) / 2 - (ylims[1] - ylims[0]) / 2 * scale_factor_y, 
                (ylims[0] + ylims[1]) / 2 + (ylims[1] - ylims[0]) / 2 * scale_factor_y)
    if lims is not None:
        plt.xlim(lims[0])
        plt.ylim(lims[1])
    elif not show_source:
        plt.xlim(new_xlims)
        plt.ylim(new_ylims)
    
    # plt.xlim(xlims)
    # plt.ylim(ylims)
    plt.legend(loc='upper left')
    plt.gca().set_aspect('equal')
    if show:
        plt.show()


    print(n_3hat)
    return
    # return axis_scalings * rot_source_I, axis_scalings * rot_panel_pixel_I


##### ESC-related stuff
def apply_convolution(data, kernel):
    # Apply 1D convolution along the second axis (horizontal, i.e. detector elements) for each projection
    return convolve1d(data, kernel, axis=1, mode='constant', cval=0.0)

def gaussian_kernel(beta, gamma, threshold=1e-3, plot=False):
    beta,gamma = float(beta),float(gamma)
    # Based on setting the gaussians equal to the threshold
    cutoff = int(np.ceil( gamma * np.sqrt(-np.log(0.5*threshold)) ))
    
    # Make sure we include the displacement in both directions
    length = round(2*cutoff + 2 * beta)
    
    # Ensure length is odd
    length = int(np.ceil(length))
    if length % 2 == 0:
        length += 1
    
    # Generate the kernel
    extend = length//2
    x = np.linspace(-extend, extend, length)
    kernel = np.exp(-((x - beta) ** 2) / (gamma ** 2)) + np.exp(-((x + beta) ** 2) / (gamma ** 2))

    if plot:
        plt.plot(x,kernel,'-o')
        plt.show()
    
    return kernel

def compute_scatter_basis(data,basis_params):
    # Currently only works for a single slice, so data should be 2-dimensional.
    # data: numpy array
    # basis params: list of lists [alpha,beta,gamma]
    S = np.zeros((len(basis_params),*data.shape))
    alpha,beta,gamma = [list(param) for param in zip(*basis_params)]
    for i in range(S.shape[0]):
        S[i] = apply_convolution(alpha[i]*data*np.exp(-data), gaussian_kernel(beta[i],gamma[i]))
    
    return S

class ESCMatrixOperator(LinearOperator):
    def __init__(self, A, domain_geometry, range_geometry, order='C'):
        """
        Custom matrix operator for ESC.

        Parameters:
        A (ndarray): The matrix to apply in the direct and adjoint operations.
        domain_geometry: VectorGeometry of the linear coefficients
        range_geometry: ImageGeometry of the linear combination of basis images
        order (str): The order of flattening and reshaping operations.
                     'C' for row-major (C-style),
                     'F' for column-major (Fortran-style).
        """
        super(ESCMatrixOperator, self).__init__(domain_geometry=domain_geometry, 
                                                range_geometry=range_geometry)
        self.A = A
        self.order = order

    def direct(self, x, out=None):
        result_1d = np.dot(self.A, x.as_array())
        result_2d = result_1d.reshape((self.range_geometry().voxel_num_y, 
                                       self.range_geometry().voxel_num_x), 
                                      order=self.order)

        if out is None:
            tmp = self.range_geometry().allocate()
            tmp.fill(result_2d)
            return tmp
        else:
            out.fill(result_2d)

    def adjoint(self, y, out=None):
        flattened_y = y.as_array().flatten(order=self.order)
        result = np.dot(self.A.T, flattened_y)

        if out is None:
            tmp = self.domain_geometry().allocate()
            tmp.fill(result)
            return tmp
        else:
            out.fill(result)
#####