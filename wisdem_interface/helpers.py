import os
import glob
import ruamel.yaml as ry
import pickle

#
# Case setup
#

# Note: if we try to load these from wisdem, it can cause MPI problems like:
#   OPAL ERROR: Unreachable in file pmix3x_client.c
# load_yaml and write_yaml are copied from wisdem/inputs/validation.py
def load_yaml(fname_input):
    reader = ry.YAML(typ="safe", pure=True)
    with open(fname_input, "r", encoding="utf-8") as f:
        input_yaml = reader.load(f)
    return input_yaml

def write_yaml(instance, foutput):
    # Write yaml with updated values
    yaml = ry.YAML()
    yaml.default_flow_style = None
    yaml.width = float("inf")
    yaml.indent(mapping=4, sequence=6, offset=3)
    yaml.allow_unicode = False
    with open(foutput, "w", encoding="utf-8") as f:
        yaml.dump(instance, f)

def load_pickle(fpath):
    with open(fpath, 'rb') as f:
        p = pickle.load(f)
    return {val['prom_name']: val for key,val in p}


def generate_tower_modeling_yaml(pkl_in,modeling_out):
    """Input pickle file should have been run with RotorSE and have
    loading data. Write out a modeling output file that allows tower
    optimization to be performed without RotorSE for much, much greater
    computational efficiency
    """
    inp = f"""\
# loading generated from {os.path.abspath(pkl_in)}
General:
    verbosity: False  # When set to True, the code prints to screen lots of info

WISDEM:
    RotorSE:
        flag: False
    TowerSE:
        flag: True
    DriveSE:
        flag: False
    BOS:
        flag: False
    n_dlc: 1
    Loading:
        mass: 999999.99999999
        center_of_mass: [0.0, 0.0, 0.0]
        moment_of_inertia: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        loads:
          - force:  [0.0, 0.0, 0.0]
            moment: [0.0, 0.0, 0.0]
            velocity: 0.0
"""
    out = load_pickle(pkl_in)
    mass   = out['drivese.rna_mass']['val'][0]
    cm     = out['drivese.rna_cm']['val']
    moi    = out['drivese.rna_I_TT']['val']
    F      = out['drivese.base_F']['val'].squeeze()
    M      = out['drivese.base_M']['val'].squeeze()
    Vrated = out['rotorse.rp.powercurve.rated_V']['val'][0]

    yaml = ry.YAML()
    yaml.default_flow_style = True
    mopts = yaml.load(inp)
    # need to explicitly cast to workaround "RepresenterError: cannot represent an object"
    mopts['WISDEM']['Loading']['mass']                 = float(mass)
    mopts['WISDEM']['Loading']['center_of_mass']       = [float(val) for val in cm]
    mopts['WISDEM']['Loading']['moment_of_inertia']    = [float(val) for val in moi]
    mopts['WISDEM']['Loading']['loads'][0]['force']    = [float(val) for val in F]
    mopts['WISDEM']['Loading']['loads'][0]['moment']   = [float(val) for val in M]
    mopts['WISDEM']['Loading']['loads'][0]['velocity'] = float(Vrated)

    with open(modeling_out, "w", encoding="utf-8") as f:
        yaml.dump(mopts, f)


#
# Analysis
#
def check_blade_freqs(steps,verbose=True):
    """Verify that blade frequencies are higher than rotor frequency
    and that the edgewise frequency is higher than the flapwise
    """
    for istep in steps:
        fpath = glob.glob(f'outputs.{istep}/*.pkl')[0]
        if verbose: print(f'Step {istep}: {fpath}')
        turb = load_pickle(fpath)
        pfx = 'comp.wt.' if istep > 1 else 'wt.'

        try:
            flapfreqs = turb[pfx+'rotorse.rs.frame.flap_mode_freqs']['value']
            edgefreqs = turb[pfx+'rotorse.rs.frame.edge_mode_freqs']['value']
        except KeyError:
            if verbose: print('No RotorSE\n')
            continue
        Omg = turb[pfx+'rotorse.rp.powercurve.compute_power_curve.rated_Omega']['value'][0]/60.
        if verbose:
            print('  flap mode freqs [Hz]:',flapfreqs)
            print('  edge mode freqs [Hz]:',edgefreqs)
            print('  3P, 6P freqs [Hz]:',3*Omg,6*Omg)
            
        # check 1st natural frequencies
        assert (flapfreqs[0] > 3*Omg), f'WARNING: 1st flap freq too low in step {istep}'
        if (flapfreqs[0] < 1.1*3*Omg):
            freq_3P = flapfreqs[0]/(3*Omg)
            print(f'WARNING: 1st flap freq does not have 10% buffer above 3P in step {istep}, ratio={freq_3P}')
        assert (edgefreqs[0] > flapfreqs[0]), f'WARNING: 1st edge freq less than 1st flap freq in step {istep}'

        # check 2nd natural frequencies
        assert (flapfreqs[1] > 6*Omg), f'WARNING: 2nd flap freq too low in step {istep}'
        if (flapfreqs[1] < 1.1*6*Omg):
            freq_6P = flapfreqs[1]/(6*Omg)
            print(f'WARNING: 2nd flap freq does not have 10% buffer above 6P in step {istep}, ratio={freq_6P}')
        assert (edgefreqs[1] > flapfreqs[1]), f'WARNING: 2nd edge freq less than 2nd flap freq in step {istep}'

        if verbose:
            print('')
        
def check_tower_freqs(steps,verbose=True):
    """Warn if tower is not soft-stiff by design"""
    Omg = None
    for istep in steps:
        fpath = glob.glob(f'outputs.{istep}/*.pkl')[0]
        if verbose: print(f'Step {istep}: {fpath}')
        turb = load_pickle(fpath)
        pfx = 'comp.wt.' if istep > 1 else 'wt.'

        try:
            Omg = turb[pfx+'rotorse.rp.powercurve.compute_power_curve.rated_Omega']['value'][0]/60.
        except KeyError:
            if verbose: print('  No RotorSE')
        else:
            if verbose: print('  1P, 3P freqs [Hz]:',1*Omg,3*Omg)
       #towerFAfreqs = turb[pfx+'towerse.post.x_mode_freqs']['value']
       #towerSSfreqs = turb[pfx+'towerse.post.y_mode_freqs']['value']
        towerFAfreqs = turb[pfx+'towerse.tower.fore_aft_freqs']['value']
        towerSSfreqs = turb[pfx+'towerse.tower.side_side_freqs']['value']
        if verbose:
            print('  tower fore-aft mode freqs [Hz]:',towerFAfreqs)
            print('  tower side-side mode freqs [Hz]:',towerSSfreqs)

        # check fore-aft natural frequencies
        assert (towerFAfreqs[0] > Omg), 'WARNING: 1st FA freq too low for soft-stiff design'
        if (towerFAfreqs[0] < 1.1*Omg):
            print(f'WARNING: 1st FA freq does not have 10% buffer above 1P in step {istep}')
        if (towerFAfreqs[0] > 3*Omg):
            print(f'WARNING: 1st FA freq too high in step {istep}')

        # check side-side natural frequencies
        assert (towerSSfreqs[0] > Omg), 'WARNING: 1st SS freq too low for soft-stiff design'
        if (towerSSfreqs[0] < 1.1*Omg):
            print(f'WARNING: 1st SS freq does not have 10% buffer above 1P in step {istep}')
        if (towerSSfreqs[0] > 3*Omg):
            print(f'WARNING: 1st SS freq too high in step {istep}')

        if verbose:
            print('')
