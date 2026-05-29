import autograd.numpy as npa
import numpy as np
import tidy3d as td
from autograd.tracer import getval
from autograd import value_and_grad
import matplotlib.pyplot as plt
import tidy3d.web as web
import optax
import json
import pickle
from pathlib import Path

# 3d specific parameters
taper_theta = npa.deg2rad(20)

# Material.
nInP = 3.13
nAir = 1.0

# Simulation wavelength.
wl = 1.55  # Central simulation wavelength (um).
bw = .04  # Simulation bandwidth (um).
n_wl = 5  # Number of wavelength points within the bandwidth.

#the parameters that kasra optimized
grating_period = .650 # period of the grating
etch_depth = 0.1 #etching depth in (um)
to_substrate = 1.210 #thickness of the substrate (um)
initial_fill_factor = 0.1 #fill factor initial for apodization
R = 0.03 #the coefficent for the apodization with units of (um^-1)
r0_extra = 4 #distance to the first tooth from the end of the waveguide (um), extra amount for taper length

# Geometric parameters.
w_thick = 0.27  # Waveguide thickness (um).
w_width = 0.55  # Waveguide width (um).
w_length = 1.0  # Waveguide length (um).
border_buffer = 0.16 # buffer around the device
N_teeth = 7 # number of teeth in the grating

#fiber output permaters all initial
source_theta = npa.deg2rad(10.0) #angle of the fiber output 
mfd = 4 #mode fiber diameter
source_x = 2.8+r0_extra #x position to the start of the first gap  
src_offset = 0.5  # Distance between the source focus and device (um).

# Material definition.
mat_InP = td.Medium(permittivity=nInP**2)  # Waveguide material.
mat_air = td.Medium(permittivity=nAir**2)  # SiO2 material.

#fabrication parameters
sidewall_angle = 0 #angle of the sidewall
dilation = 0 #dilation of the sidewalls

# Wavelengths and frequencies.
wl_max = wl + bw / 2
wl_min = wl - bw / 2
wl_range = npa.linspace(wl_min, wl_max, n_wl)
freq = td.C_0 / wl
freqs = td.C_0 / wl_range
freqw = 0.5 * (freqs[0] - freqs[-1])
run_time = 2e-12

# Computational domain size.
pml_spacing = 0.6 * wl
inf_eff = 1000
min_steps_per_wvl = 20

#define the bounds for the parameters
widths_bounds = (0.05, 0.6)
r0_bounds = (4,7)
etch_depth_bounds = (0.05, 0.2)
to_substrate_bounds = (.5,2)

def projection_builder(widths_bounds=widths_bounds, 
    r0_bounds=r0_bounds, 
    etch_depth_bounds=etch_depth_bounds, 
    to_substrate_bounds=to_substrate_bounds,
    N_teeth=N_teeth
):
    """
    This function builds the projection and inverse projection functions for a given set of bounds.
    """
    maxes = npa.array([*([widths_bounds[1]]*(N_teeth*2+2)), r0_bounds[1], etch_depth_bounds[1], to_substrate_bounds[1]])
    mins = npa.array([*([widths_bounds[0]]*(N_teeth*2+2)), r0_bounds[0], etch_depth_bounds[0], to_substrate_bounds[0]])

    def project(params):
        return 0.5*(maxes-mins)*npa.tanh(params)+0.5*(maxes+mins)
    
    def inverse_project(params):
        return npa.arctanh((2*params-(maxes+mins))/(maxes-mins))
    
    return project, inverse_project

def get_centers(widths, r0):
    """
    This function takes in a list of widths and returns a list of centers.

    Args:
        widths (list): A list of widths.
        r0 (float): The radius that the grating starts at.

    Returns:
        list: A list of centers.
    """
    centers = []
    for i in range(len(widths)):
        centers.append(r0+npa.sum(widths[:i])+widths[i]/2)
    return centers

def get_tooth_arc(r,w_width=w_width,taper_theta=taper_theta,theta_div=150):
    """
    This resurns the xy values for the tooth arc
    """
    thetas = npa.linspace(taper_theta,-taper_theta,theta_div,endpoint=True)
    D = w_width/2/np.tan(taper_theta)
    xs = (D+r)*npa.cos(thetas)-D
    ys = (D+r)*npa.sin(thetas)
    return npa.column_stack((xs,ys))


def make_grating_structure(
    widths,
    etch_depth=etch_depth,
    to_substrate=to_substrate,
    r0=r0_extra,
    N_teeth=N_teeth,
    grating_period=grating_period,
    w_thick=w_thick,
    border_buffer=border_buffer,
    sidewall_angle=sidewall_angle,
    dilation=dilation,
):
    """
    This function makes the grating structure.

    Args:
        widths (list): A list of widths.
        etch_depth (float): The etching depth.
        to_substrate (float): The thickness of the substrate.
        r0 (float): The radius that the grating starts at.
        N_teeth (int): The number of teeth in the grating.
        sidewall_angle (float): The angle of the sidewall.
        dilation (float): The dilation of the sidewalls.
    """
    #get the centers of the teeth and gaps
    centers = get_centers(widths, r0)

    #untracked values
    dilation_untracked = float(getval(dilation))
    etch_depth_untracked = float(getval(etch_depth))

    #create the substrate
    substrate = td.Structure(
            geometry=td.Box.from_bounds((-inf_eff,-inf_eff,-inf_eff),(inf_eff,inf_eff,-to_substrate)),
            medium=mat_InP,
            name='substrate')

    #create the waveguide
    waveguide = td.Structure(
        geometry=td.Box.from_bounds((-inf_eff,-w_width/2,0),
                                    (0,w_width/2,w_thick)),
        medium=mat_InP,
        name='bottom_waveguide')
    
    #create taper 
    x_taper = npa.linspace(0, (r0+dilation+w_width/2/npa.tan(taper_theta))*npa.cos(taper_theta)-w_width/2/npa.tan(taper_theta), 500)
    y_taper = npa.linspace(w_width/2, (w_width/2/npa.tan(taper_theta)+r0+dilation)*npa.sin(taper_theta), 500)
    taper_vertices = npa.concatenate([
        npa.column_stack((x_taper[:-1], y_taper[:-1])),
        get_tooth_arc(r0+dilation),
        npa.column_stack((x_taper[:-1][::-1], -y_taper[:-1][::-1]))])
    
    top_taper = td.Structure(
        geometry=td.PolySlab(
        vertices=taper_vertices,
        axis=2,
        slab_bounds=(w_thick-etch_depth_untracked,w_thick)),
        medium=mat_InP,
        name='top_taper')

    bottom_taper = td.Structure(
        geometry=td.PolySlab(
        vertices=npa.array(taper_vertices),
        axis=2,
        slab_bounds=(0,w_thick-etch_depth_untracked)),
        medium=mat_InP,
        name='bottom_taper')

    #create the end waveguide
    end_arc = get_tooth_arc(centers[-1]-widths[-1]/2-dilation)
    end_vertices = npa.concatenate([
        end_arc,
        npa.column_stack((inf_eff,end_arc[-1,1])),
        npa.column_stack((inf_eff,end_arc[0,1])),
    ])
    end_waveguide = td.Structure(   
        geometry=td.PolySlab(
        vertices=end_vertices,
        axis=2,
        slab_bounds=(0,w_thick)),
        medium=mat_InP,
        name='top_end_waveguide')

    #get the centers of the teeth and gaps
    teeth = []
    teeth_low = []
    teeth_gap_low = []
    for i, (center, width) in enumerate(zip(centers, widths)):

        if i % 2 == 0:
            vertices = npa.concatenate([
                get_tooth_arc(center-width/2+dilation),
                get_tooth_arc(center+width/2-dilation)[::-1],
            ])
            tooth_geom_gap_low = td.PolySlab(
                vertices=getval(vertices),
                axis=2,
                slab_bounds=(0,w_thick-etch_depth))
            teeth_gap_low.append(tooth_geom_gap_low)

        if i%2 == 1 and i != len(centers)-1:
            vertices = npa.concatenate([
                get_tooth_arc(center-width/2-dilation),
                get_tooth_arc(center+width/2+dilation)[::-1],
            ])
            tooth_geom = td.PolySlab(
                vertices=vertices,
                axis=2,
                slab_bounds=(w_thick-etch_depth_untracked,w_thick),
                sidewall_angle=sidewall_angle)
            teeth.append(tooth_geom)

            tooth_geom_low = td.PolySlab(
                vertices=getval(vertices),
                axis=2,
                slab_bounds=(0,w_thick-etch_depth_untracked))
            teeth_low.append(tooth_geom_low)
            
    teeth = td.Structure(
            geometry=td.GeometryGroup(geometries=teeth),
            medium=mat_InP,
            name='teeth')
    teeth_low = td.Structure(
            geometry=td.GeometryGroup(geometries=teeth_low),
            medium=mat_InP,
            name='teeth_low')
    teeth_gap_low = td.Structure(
            geometry=td.GeometryGroup(geometries=teeth_gap_low),
            medium=mat_InP,
            name='teeth_gap_low')

    #define the minumum and maximum x and z bounds
    min_x = -border_buffer-pml_spacing
    max_x = border_buffer+pml_spacing+getval(centers[-1]+widths[-1]/2+dilation)
    min_y = getval(end_arc[-1,1])-pml_spacing
    max_y = getval(end_arc[0,1])+pml_spacing
    min_z = -border_buffer-pml_spacing-getval(to_substrate)
    max_z = w_thick+border_buffer+pml_spacing

    sim_box = td.Box.from_bounds(
        rmin=(min_x,min_y,min_z),
        rmax=(max_x,max_y,max_z),
    )

    structures = [substrate, waveguide, top_taper, bottom_taper, teeth, teeth_gap_low, teeth_low, end_waveguide]
    return structures, sim_box

def make_sim(
    widths,
    etch_depth=etch_depth,
    to_substrate=to_substrate,
    grating_period=grating_period,
    r0=r0_extra,
    N_teeth=N_teeth,
    w_thick=w_thick,
    border_buffer=border_buffer,
    sidewall_angle=sidewall_angle,
    dilation=dilation,
    source_x=source_x,
    source_theta=source_theta,
    mfd=mfd,
    src_offset=src_offset,
    include_field_monitor=False,
    freq=freq,
    freqw=freqw,
    freqs=freqs,
    run_time=run_time,
    pml_spacing=pml_spacing,
    min_steps_per_wvl=min_steps_per_wvl,
):  

    """ 
    This function makes the simulation.
    """

    #get the structures
    structures,sim_box = make_grating_structure(widths,
                etch_depth=etch_depth,
                to_substrate=to_substrate,
                r0=r0,
                N_teeth=N_teeth,
                grating_period=grating_period,
                w_thick=w_thick,
                border_buffer=border_buffer,
                sidewall_angle=sidewall_angle,
                dilation=dilation,
                )

    #define the gaussian beam source
    sources = [td.GaussianBeam(
        size=(2 * mfd, 2 * mfd, 0),
        center=[source_x, 0, w_thick+src_offset],
        source_time=td.GaussianPulse(freq0=freq, fwidth=freqw),
        angle_theta=source_theta,
        direction="-",
        waist_radius=mfd / 2,
        pol_angle=np.pi / 2,  # 90 degree polarization angle for TE polarization
        waist_distance=-src_offset,
    )]

    #define the mode monitor
    monitors = [
        td.ModeMonitor(
        center=(-border_buffer, 0, w_thick / 2),
        size=(0, w_width*2, w_thick*4),
        freqs=freqs,
        mode_spec=td.ModeSpec(num_modes=1, target_neff=nInP),
        name="mode",
    )]

    #if we want to include the field monitor
    if include_field_monitor:
        monitors.append(
            td.FieldMonitor(
                center=(0, 0, 0),
                size=(inf_eff, inf_eff, inf_eff),
                freqs=freq,
                name="field",
            )
        )

    sim = td.Simulation(
        center=sim_box.center,
        size=sim_box.size,
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=min_steps_per_wvl),
        structures=structures,
        sources=sources,
        monitors=monitors,
        run_time=run_time,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(),
            y=td.Boundary.pml(),
            z=td.Boundary.pml(),
        ),
        symmetry=(0,-1,0),
    )
    return sim

def get_coupling_efficiency(sim_data):
    """
    This function takes in the simulation data and returns the coupling efficiency.
    """
    return np.abs(sim_data["mode"].amps.sel(mode_index=0,direction="-").values)**2


#---------------------------------
#--Saving and Loading Functions--####
#---------------------------------

def save_checkpoint(step, params, opt_state, J, grad, filename="data/3d_opt/tmp.pkl"):
    data = {
        "step": step,
        "params": params,
        "opt_state": opt_state,
        "J": J,
        "grad": grad
    }

    # ensure directory exists
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_checkpoint(filename="data/3d_opt/tmp.pkl"):
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return data["step"], data["params"], data["opt_state"], data["J"], data["grad"]
    except FileNotFoundError:
        return None

###---------------------------------
###--Runs Adam Optimization--####
###---------------------------------

def run_adam(params0,
    projection,
    inverse_projection,
    objective,
    num_steps=25,
    learning_rate=0.01,
    verbose=True,
    opt_state=None,
    history=None,
    savepath='data/3d_opt/tmp.pkl'):
    """
    This function runs the Adam optimization.
    """
    params = inverse_projection(params0)
    optimizer = optax.adam(learning_rate=learning_rate)
    if opt_state is None:
        opt_state = optimizer.init(params)

    #store the history
    if history is None:
        history = {
        "params": [],
        "J": [],
        "grad_norm": [],
        "grad": []
        }

    dj = value_and_grad(objective)

    for i in range(num_steps):
        value, gradient = dj(params,projection=projection)
        updates, opt_state = optimizer.update(-gradient, opt_state, params)
        params[:] = optax.apply_updates(params, updates)
        history["params"].append(projection(params).copy().tolist())
        history["J"].append(float(value))
        history["grad_norm"].append(float(np.linalg.norm(gradient)))
        history["grad"].append(gradient.copy().tolist())
        if verbose:
            print(f"step = {i + 1}")
            print(f"\tJ = {value:.3e}")
            print(f"\tgrad_norm = {np.linalg.norm(gradient):.4e}")

            #print angle between the gradients
            if i!=0:
                print(f"\tangle = {np.arccos(np.dot(gradient, history['grad'][-2])/(np.linalg.norm(gradient)*np.linalg.norm(history['grad'][-2])))*180/np.pi}")

        save_checkpoint(i, projection(params), opt_state, value, gradient, filename=savepath)


    return history, opt_state


