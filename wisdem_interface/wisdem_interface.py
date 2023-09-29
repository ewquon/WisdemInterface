import os
import sys
import subprocess
import copy
import time

from wisdem_interface.helpers import load_yaml, save_yaml # legacy functions
#import wisdem.inputs as schema
from wisdem.glue_code.runWISDEM import load_wisdem
import wisdem.postprocessing.wisdem_get as getter


class WisdemInterface(object):
    """An interface that will automatically generate runscript and input
    files as needed, and then call WISDEM
    """
    def __init__(self,
                 turbine_prefix,
                 starting_geometry,
                 default_modeling_options,
                 default_analysis_options,
                 run_dir='.',
                 outdir_prefix='design',
                 runscript_prefix='run_wisdem',
                 mpirun='mpirun',
                 tol=1e-4):
        self.run_dir = run_dir
        self.prefix = turbine_prefix
        self.outdir_prefix = outdir_prefix
        self.runscript_prefix = runscript_prefix
        self.mpirun = mpirun

        self.wt_opt = None
        self.mopt = load_yaml(default_modeling_options) # TODO use schema
        self.aopt = load_yaml(default_analysis_options)

        try:
            self.maxranks = int(os.environ['SLURM_NTASKS'])
        except KeyError:
            self.maxranks = 1
        print('Maximum number of MPI ranks =',self.maxranks)

        self.optstep = 0
        self.optlabels = []
        self.outfpaths = []
        self.aopt['driver']['optimization']['flag'] = False
        self.optimize(label='Baseline',geom_path=starting_geometry)

        # save to quickly reset later
        self.aopt['driver']['optimization']['flag'] = True
        self.aopt['driver']['optimization']['tol'] = tol
        self.aopt_baseline = copy.deepcopy(self.aopt)
        self.mopt_baseline = copy.deepcopy(self.mopt)


    def reset(self,analysis_options=True,modeling_options=True):
        if analysis_options:
            print('\nResetting analysis options')
            self.aopt = copy.deepcopy(self.aopt_baseline)
        if modeling_options:
            print('\nResetting modeling options')
            self.mopt = copy.deepcopy(self.mopt_baseline)


    def _write_inputs_and_runscript(self,
                                    fpath_wt_input,
                                    fpath_modeling_options,
                                    fpath_analysis_options,
                                    model_changes={}):
        """Write out input files and a runscript that may be called in
        parallel. A bit roundabout, but might be easier than trying to
        figure out how to invoke MPI internally for a different number
        of CPU cores per design step.
        """
        save_yaml(fpath_analysis_options, self.aopt)
        save_yaml(fpath_modeling_options, self.mopt)
        runscript = os.path.join(
                self.run_dir, f'{self.runscript_prefix}.{self.optstep}.py')
        with open(runscript,'w') as f:
            f.write(f'''from wisdem import run_wisdem

wt_opt, modeling_options, opt_options = run_wisdem(
    '{fpath_wt_input}',
    '{fpath_modeling_options}',
    '{fpath_analysis_options}',
    overridden_values={str(model_changes)}
)''')
        return runscript


    def write_postproc_script(self,fpath=None):
        if fpath is None:
            fpath = f'compare_{self.prefix}_designs.sh'
        labels = [f'"{label}"' for label in self.optlabels]
        cmd = 'compare_designs \\\n\t' \
            + ' \\\n\t'.join(self.outfpaths) \
            + ' \\\n\t--labels ' \
            + ' '.join(labels)
        with open(fpath,'w') as f:
            f.write('grep "^Optimization" log.wisdem.?\n\n')
            f.write(cmd)
        print('\nWrote postprocessing script',fpath)


    def _get_num_finite_differences(self):
        if self.optstep == 0:
            if self.aopt['driver']['optimization']['flag']:
                print('Should not be optimizing in baseline run')
            return 0

        if self.aopt['driver']['optimization']['form'] == 'forward':
            dv_fac = 1
        elif self.aopt['driver']['optimization']['form'] == 'central':
            dv_fac = 2
        else:
            raise NotImplementedError('Unexpected finite differencing mode')

        # figure out number of finite differences for running in parallel
        to_opt = []
        n_fd = 0
        for prop in self.aopt['design_variables']['blade']['aero_shape']:
            # e.g., twist, chord
            optctrl = self.aopt['design_variables']['blade']['aero_shape'][prop]
            if optctrl['flag'] == True:
                to_opt.append(f'blade:aero_shape:{prop}')
                try:
                    # optimize control points from index_start to index_end-1
                    n_opt = optctrl['index_end'] - optctrl['index_start']
                except KeyError:
                    # range not specified, optimizing all control points
                    n_opt = optctrl['n_opt']
                n_fd += dv_fac * n_opt

        for prop in self.aopt['design_variables']['blade']['structure']:
            # e.g., spar_cap_*
            optctrl = self.aopt['design_variables']['blade']['structure'][prop]
            if optctrl['flag'] == True:
                if prop == 'spar_cap_ps':
                    # handle special case of pressure-side spar-cap thickness
                    # equal to the suction side -- don't need to add extra
                    # control points
                    equal_to_suction = optctrl.get('equal_to_suction',False)
                    if equal_to_suction:
                        print('Pressure-side spar-cap thickness '
                              'equal to suction side')
                        if 'n_opt' in optctrl.keys():
                            print('Ignoring spar_cap_ps.n_opt =',
                                  optctrl['n_opt'])
                        continue

                to_opt.append(f'blade:structure:{prop}')
                try:
                    # optimize control points from index_start to index_end-1
                    n_opt = optctrl['index_end'] - optctrl['index_start']
                except KeyError:
                    # range not specified, optimizing all control points
                    n_opt = optctrl['n_opt']
                n_fd += dv_fac * n_opt

        for prop in self.aopt['design_variables']['blade']['dac']:
            # e.g., te_flap_*
            optctrl = self.aopt['design_variables']['blade']['dac'][prop]
            if optctrl['flag'] == True:
                to_opt.append(f'blade:dac:{prop}')
                n_fd += dv_fac

        optctrl = self.aopt['design_variables']['control']['tsr']
        if optctrl['flag'] == True:
            to_opt.append(f'control:tsr')
            n_fd += dv_fac

        for prop in self.aopt['design_variables']['control']['servo']:
            # e.g., torque_control
            optctrl = self.aopt['design_variables']['control']['servo'][prop]
            if optctrl['flag'] == True:
                to_opt.append(f'control:servo:{prop}')
                n_fd += dv_fac

        for prop in self.aopt['design_variables']['tower']:
            # e.g., outer_diameter
            optctrl = self.aopt['design_variables']['tower'][prop]
            if optctrl['flag'] == True:
                to_opt.append(f'tower:{prop}')
                if prop == 'outer_diameter':
                    grid = getter.get_tower_diameter(self.wt_opt)
                elif prop == 'layer_thickness':
                    grid = getter.get_tower_thickness(self.wt_opt)
                elif prop == 'section_height':
                    grid = getter.get_section_height(self.wt_opt)
                n_opt = len(grid)
                n_fd += dv_fac * n_opt

        print('Optimizations:',to_opt)
        print('Number of finite differences needed:',n_fd)
        return n_fd


    def _execute(self, runscript=None, serial=False):
        if runscript is None:
            runscript = f'{self.runscript_prefix}.{self.optstep}.py'
        cmd = ['python',runscript]

        try_mpi = (not serial) and (self.maxranks > 1)
        if try_mpi:
            n_fd = self._get_num_finite_differences()
            nranks = min(max(1,n_fd), self.maxranks)
            if n_fd > 0:
                cmd = [self.mpirun,'-n',f'{nranks:d}'] + cmd

        print('Executing:',' '.join(cmd))
        with open(f'log.wisdem.{self.optstep}','w') as log:
            subprocess.run(
                    cmd, stdout=log, stderr=subprocess.STDOUT, text=True)

        
    def optimize(self, label=None, geom_path=None, rerun=False, serial=False):
        if label is None:
            label = f'Opt step {self.optstep}'
        self.optlabels.append(label)

        # input file for step > 0 comes from previous output dir
        if self.optstep == 0:
            print('\n=== Running WISDEM baseline case ===')
            assert geom_path is not None
            wt_input = geom_path
            wt_output = f'{self.prefix}-step0.yaml'
        else:
            print(f'\n=== Running optimization step {self.optstep}: {label} ===')
            wt_input = os.path.join(f'{self.outdir_prefix}.{self.optstep-1}',
                                    f'{self.prefix}-step{self.optstep-1}.yaml')
            wt_output = f'{self.prefix}-step{self.optstep}.yaml'

        # create new output dir
        outdir = os.path.join(self.run_dir, f'{self.outdir_prefix}.{self.optstep}')
        os.makedirs(outdir, exist_ok=True)
        self.aopt['general']['folder_output'] = outdir
        self.aopt['general']['fname_output'] = wt_output

        # save output file path to put into postproc script later
        full_wt_output_path = os.path.join(outdir, wt_output)
        self.outfpaths.append(full_wt_output_path)

        # put analysis and modeling inputs in output dir so that
        # everything works with the WISDEM compare_designs script
        fpath_wt_input = os.path.join(self.run_dir, wt_input)
        fpath_modeling_options = os.path.join(outdir,
                f'{self.prefix}-step{self.optstep}-modeling.yaml')
        fpath_analysis_options = os.path.join(outdir,
                f'{self.prefix}-step{self.optstep}-analysis.yaml')

        # don't overwrite inputs/outputs unless rerun=True
        if (not os.path.isfile(full_wt_output_path)) or rerun:
            runscript = self._write_inputs_and_runscript(
                    fpath_wt_input, fpath_modeling_options, fpath_analysis_options)
            tt = time.time()
            self._execute(runscript, serial=serial)
            print('Run time: %f'%(time.time()-tt))
            sys.stdout.flush()
        else:
            print(full_wt_output_path,'found,'
                  ' set rerun=True to repeat this optimization step')

        self._post_opt_actions(outdir)

        self.optstep += 1


    def _post_opt_actions(self,outdir):
        """At this point, we finished an optimization but the optstep
        has not been incremented yet.
        """
        #self.verify_converged() # TODO
        wt_output = os.path.join(outdir,
                                 f'{self.prefix}-step{self.optstep}.yaml')
        if not os.path.isfile(wt_output):
            sys.exit(f'Problem with optimization step {self.optstep}, '
                     f'{wt_output} not found')

        # load new turbine data object, an instance of
        # openmdao.core.problem.Problem
        self.wt_opt, _, _ = load_wisdem(wt_output)

        # update loading, to be used in TowerSE-only analysis, if needed
        self.get_rna_loading()


    def get_rna_loading(self):
        """Get loading on tower from rotor-nacelle assembly

        Assume n_dlc==1
        """
        if not self.mopt['WISDEM']['TowerSE']['flag']:
            print('TowerSE not active, no RNA loading available')
            self.rna_loading = None
            return

        # need to explicitly cast to float -- np.ndarray.astype doesn't work --
        # as a workaround to what appears to be the issue here:
        # https://github.com/SimplyKnownAsG/yamlize/issues/3
        rna_mass = float(self.wt_opt['towerse.rna_mass'][0])
        rna_cg = [float(val) for val in self.wt_opt['towerse.rna_cg']]
        rna_I = [float(val) for val in self.wt_opt['towerse.rna_I']]
        rna_F = [float(val) for val in self.wt_opt['towerse.tower.rna_F'].squeeze()]
        rna_M = [float(val) for val in self.wt_opt['towerse.tower.rna_M'].squeeze()]
        if self.mopt['WISDEM']['RotorSE']['flag']:
            Vrated = float(self.wt_opt['rotorse.rp.powercurve.rated_V'][0])
        else:
            Vrated = self.rna_loading['loads'][0]['velocity']
            print('RotorSE deactivated, keeping Vrated =',Vrated,' unchanged')

        self.rna_loading = {
            'mass': rna_mass,
            'center_of_mass': rna_cg,
            'moment_of_inertia': rna_I,
            'loads': [
                # a list of load cases (default: n_dlc=1)
                {'force': rna_F, 'moment': rna_M, 'velocity': Vrated},
            ],
        }
