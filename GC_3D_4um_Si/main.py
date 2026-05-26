import autograd.numpy as np
import tidy3d as td
from autograd.tracer import getval
from autograd import value_and_grad
import optax
from pathlib import Path
import pickle

# Material.
nSi = 3.4487
nSiO2 = 1.4431
nAir = 1.0

# Simulation wavelength.
wl = 1.55  # Central simulation wavelength (um).
bw = .04  # Simulation bandwidth (um).
n_wl = 5  # Number of wavelength points within the bandwidth.

#the parameters that kasra optimized
grating_period = .650 # period of the grating
to_substrate = 2 #thickness of the substrate (um)
initial_fill_factor = 0.16 #fill factor initial for apodization
R = 0.03 #the coefficent for the apodization with units of (um^-1)
r0 = 1 #distance to the first tooth from the end of the waveguide (um)

# Geometric parameters.
w_thick = 0.22  # Waveguide thickness (um).
w_width = 0.4  # Waveguide width (um).
w_length = 1.0  # Waveguide length (um).
border_buffer = 0.16 # buffer around the device
N_teeth = 9 # number of teeth in the grating

#fiber output permaters all initial
source_theta = np.deg2rad(5.0) #angle of the fiber output 
mfd = 4 #mode fiber diameter
source_x = 4+2.5 #x position of the fiber 
src_offset = 2.7  # Distance between the source focus and device (um).
focus_distance = 1.25 #distance between the source and the focus (um)

# Material definition.
mat_Si = td.Medium(permittivity=nSi**2)  # Waveguide material.
mat_SiO2 = td.Medium(permittivity=nSiO2**2)  # SiO2 material.
mat_air = td.Medium(permittivity=nAir**2)  # Air material.

#fabrication parameters
sidewall_angle = 0 #angle of the sidewall
dilation = 0 #dilation of the sidewalls
taper_theta = np.deg2rad(20)

# Wavelengths and frequencies.
wl_max = wl + bw / 2
wl_min = wl - bw / 2
wl_range = np.linspace(wl_min, wl_max, n_wl)
freq = td.C_0 / wl
freqs = td.C_0 / wl_range
freqw = 0.5 * (freqs[0] - freqs[-1])
run_time = 1e-12

# Computational domain size.
pml_spacing = 0.6 * wl
inf_eff = 1000
min_steps_per_wvl = 20

#define the bounds for the parameters
widths_bounds = (0.075, 0.6)
r0_bounds = (1,6)

def projection_builder(widths_bounds=widths_bounds, 
    r0_bounds=r0_bounds, 
    N_teeth=N_teeth
):
    """
    This function builds the projection and inverse projection functions for a given set of bounds.
    """
    maxes = np.array([*([widths_bounds[1]]*(N_teeth*2+2)), r0_bounds[1]])
    mins = np.array([*([widths_bounds[0]]*(N_teeth*2+2)), r0_bounds[0]])

    def project(params):
        return 0.5*(maxes-mins)*np.tanh(params)+0.5*(maxes+mins)
    
    def inverse_project(params):
        return np.arctanh((2*params-(maxes+mins))/(maxes-mins))
    
    return project, inverse_project

def apodized_to_widths(R=R, initial_fill_factor=initial_fill_factor, N_teeth=N_teeth, grating_period=grating_period):
    """
    This function take the apodization variables and return a list of widths.

    Args:
        r0 (float): The radius that the grating starts at.
        R (float): The rate of the apodization.
        initial_fill_factor (list): A list of initial fill factors.

    Returns:
        list: A list of widths.
    """
    widths = []
    for i in range(N_teeth):
        ff = initial_fill_factor + R * i * grating_period
        widths.append(grating_period * ff)
        widths.append(grating_period * (1-ff))
    widths.append(grating_period * ff)
    return widths
    
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
        centers.append(r0+np.sum(widths[:i])+widths[i]/2)
    return centers

def get_tooth_arc(r,w_width=w_width,taper_theta=taper_theta,theta_div=150):
    """
    This resurns the xy values for the tooth arc
    """
    thetas = np.linspace(taper_theta,-taper_theta,theta_div,endpoint=True)
    D = w_width/2/np.tan(taper_theta)
    xs = (D+r)*np.cos(thetas)-D
    ys = (D+r)*np.sin(thetas)
    return np.column_stack((xs,ys))

# ----- structure and simulation builders -----
def make_grating_structure(
    widths,
    to_substrate=to_substrate,
    r0=r0,
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
        
        to_substrate (float): The thickness of the substrate.
        r0 (float): The radius that the grating starts at.
        N_teeth (int): The number of teeth in the grating.
        sidewall_angle (float): The angle of the sidewall.
        dilation (float): The dilation of the sidewalls.
    """
    #get the centers of the teeth and gaps
    centers = get_centers(widths, r0)

    #create the substrate
    substrate = td.Structure(
            geometry=td.Box.from_bounds((-inf_eff,-inf_eff,-inf_eff),(inf_eff,inf_eff,-to_substrate)),
            medium=mat_Si,
            name='substrate')

    #create the start waveguide
    start_waveguide = td.Structure(
        geometry=td.Box.from_bounds((-inf_eff,-w_width/2,0),
                                    (0,w_width/2,w_thick)),
        medium=mat_Si,
        name='start_waveguide')

    #create the end waveguide
    end_arc = get_tooth_arc(centers[-1]+widths[-1]/2-dilation)
    end_vertices = np.concatenate([
        end_arc,
        np.column_stack((inf_eff,end_arc[-1,1])),
        np.column_stack((inf_eff,end_arc[0,1])),
    ])
    end_waveguide = td.Structure(   
        geometry=td.PolySlab(
        vertices=end_vertices,
        axis=2,
        slab_bounds=(0,w_thick)),
        medium=mat_Si,
        name='top_end_waveguide')

    #create taper 
    x_taper = np.linspace(0, (r0+dilation+w_width/2/np.tan(taper_theta))*np.cos(taper_theta)-w_width/2/np.tan(taper_theta), 500)
    y_taper = np.linspace(w_width/2, (w_width/2/np.tan(taper_theta)+r0+dilation)*np.sin(taper_theta), 500)
    taper_vertices = np.concatenate([
        np.column_stack((x_taper[:-1], y_taper[:-1])),
        get_tooth_arc(r0+dilation),
        np.column_stack((x_taper[:-1][::-1], -y_taper[:-1][::-1]))])
    
    taper = td.Structure(
        geometry=td.PolySlab(
        vertices=taper_vertices,
        axis=2,
        slab_bounds=(0,w_thick)),
        medium=mat_Si,
        name='top_taper')


    #create the air above 
    air = td.Structure(
        geometry=td.Box.from_bounds((-inf_eff,-inf_eff,2.2),
                                    (inf_eff,inf_eff,inf_eff)),
        medium=mat_air,
        name='air')

    #get the centers of the teeth and gaps
    teeth = []
    for i, (center, width) in enumerate(zip(centers, widths)):

        if i % 2 == 1:
            vertices = np.concatenate([
                get_tooth_arc(center-width/2+dilation),
                get_tooth_arc(center+width/2-dilation)[::-1],
            ])
            tooth_geom_gap = td.PolySlab(
                vertices=vertices,
                axis=2,
                slab_bounds=(0,w_thick))
            teeth.append(tooth_geom_gap)
            
    teeth = td.Structure(
            geometry=td.GeometryGroup(geometries=teeth),
            medium=mat_Si,
            name='teeth')

    #define the minumum and maximum x and z bounds
    min_x = -border_buffer-pml_spacing
    max_x = border_buffer+pml_spacing+getval(centers[-1])+getval(widths[-1])/2+getval(dilation)
    min_y = getval(end_arc[-1,1])-border_buffer-pml_spacing
    max_y = getval(end_arc[0,1])+border_buffer+pml_spacing
    min_z = -border_buffer-pml_spacing-getval(to_substrate)
    max_z = 2.2+border_buffer+pml_spacing

    sim_box = td.Box.from_bounds(
        rmin=(min_x,min_y,min_z),
        rmax=(max_x,max_y,max_z),
    )

    structures = [substrate, start_waveguide, end_waveguide, taper, teeth, air]
    return structures, sim_box

def make_sim(
    widths,
    to_substrate=to_substrate,
    grating_period=grating_period,
    r0=r0,
    N_teeth=N_teeth,
    w_thick=w_thick,
    border_buffer=border_buffer,
    sidewall_angle=sidewall_angle,
    dilation=dilation,
    source_x=source_x,
    source_y=0,
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
    focus_distance=focus_distance,
):  

    """ 
    This function makes the simulation.
    """

    #get the structures
    structures,sim_box = make_grating_structure(widths,
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
        center=[source_x, source_y, w_thick+src_offset],
        source_time=td.GaussianPulse(freq0=freq, fwidth=freqw),
        angle_theta=source_theta,
        direction="-",
        waist_radius=mfd / 2,
        pol_angle=np.pi / 2,  # 90 degree polarization angle for TE polarization
        waist_distance=-focus_distance-src_offset,
    )]

    #define the mode monitor
    monitors = [
        td.ModeMonitor(
        center=(-2*border_buffer, 0, w_thick / 2),
        size=(0, w_width*4, w_thick*4),
        freqs=freqs,
        mode_spec=td.ModeSpec(num_modes=1, target_neff=nSi),
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

    if source_y == 0:
        symmetry = (0, -1, 0)
    else:
        symmetry = (0,0,0)
        
    sim = td.Simulation(
        center=sim_box.center,
        size=sim_box.size,
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=min_steps_per_wvl),
        structures=structures,
        sources=sources,
        monitors=monitors,
        run_time=run_time,
        medium=mat_SiO2,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(),
            y=td.Boundary.pml(),
            z=td.Boundary.pml(),
        ),
        symmetry=symmetry
    )
    return sim

def get_coupling_efficiency(sim_data):
    """
    This function takes in the simulation data and returns the coupling efficiency.
    """
    return np.abs(sim_data["mode"].amps.sel(mode_index=0,direction="-").values)**2


# ----- saving and loading -----
def save_checkpoint(step, params, opt_state, J, grad, history, filename="data/opt/tmp.pkl"):
    data = {
        "step": step,
        "params": params,
        "opt_state": opt_state,
        "J": J,
        "grad": grad,
        "history": history
    }

    # ensure directory exists
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_checkpoint(filename="data/opt/tmp.pkl"):
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return data["step"], data["params"], data["opt_state"], data["J"], data["grad"], data["history"]
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

        save_checkpoint(i, projection(params), opt_state, value, gradient, history, filename=savepath)


    return history, opt_state

