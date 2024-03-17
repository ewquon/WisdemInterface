import os
import sys
import shutil
import subprocess
import copy
from pprint import pprint
import time

from wisdem_interface.helpers import load_yaml, write_yaml, load_pickle


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
                 outdir_prefix='wisdem',
                 runscript_prefix='run_wisdem',
                 mpirun='mpirun',
                 tol=1e-4):
        self.run_dir = run_dir
        self.prefix = turbine_prefix
        self.outdir_prefix = outdir_prefix
        self.runscript_prefix = runscript_prefix
        self.mpirun = mpirun

        self.wt_opt = None
        self.geom = None # set after running optimize
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

    def add_blade_struct_dv(self,
                            layer_name,
                            n_opt=8,
                            max_decrease=0.5,
                            max_increase=1.5,
                            index_start=0,
                            index_end=8,
                            **kwargs):
        """Helper function to add blade structural optimization design
        variables. Makes use of functionality available in WISDEM
        v3.13.x after PR#490. 
        """
        layers = self.geom['components']['blade']['internal_structure_2d_fem']['layers']
        assert layer_name in [layer['name'] for layer in layers], \
                f'{layer_name} not found for blade structural optimization'
        dvs = self.aopt['design_variables']['blade']['structure']
        for dv in dvs:
            if layer_name == dv['layer_name']:
                print('Design variables already associated with',layer_name)
                return dv
        newdv = dict(layer_name=layer_name,
                     n_opt=n_opt,
                     max_decrease=max_decrease,
                     max_increase=max_increase,
                     index_start=index_start,
                     index_end=index_end)
        for key,val in kwargs.items():
            if key in newdv.keys():
                print(f'Ignoring kwarg "{key}", key already specified')
            else:
                newdv[key] = val
        dvs.append(newdv)
        return newdv

    def get_blade_layers(self,*names):
        """Get blade internal layer definition by name(s)"""
        wt_input = os.path.join(f'{self.outdir_prefix}.{self.optstep-1}',
                                f'{self.prefix}-step{self.optstep-1}.yaml')
        geom = load_yaml(wt_input)
        layers = geom['components']['blade']['internal_structure_2d_fem']['layers']
        defn = {}
        for name in names:
            for layer in layers:
                if layer['name'] == name:
                    defn[name] = layer
        return defn

    def modify_blade_layers(self,geom_path=None,**kwargs):
        """This is an EXPERIMENTAL feature -- use at your own risk!

        Call this prior to running optimize() to modify the geometry
        from a previous step. The kwargs are a mapping between layer
        names and layer definition dictionaries. If a geom_path is not
        specified, then a backup will be made in the same directory as
        the starting geometry file.
        """
        # this is the input file that will be used for the next optimization
        wt_input = os.path.join(f'{self.outdir_prefix}.{self.optstep-1}',
                                f'{self.prefix}-step{self.optstep-1}.yaml')
        if geom_path is None:
            backup = wt_input+'.original'
            assert not os.path.isfile(backup), \
                    ('Geometry has already been modified?! '
                     'modify_geometry() should not have been called more than once '
                     'per step')
            shutil.copyfile(wt_input, backup)
        elif not os.path.isabs(geom_path):
            # make paths relative to previous optimization dir where the
            # starting geometry for the next optmization step typically lives
            geom_path = os.path.join(f'{self.outdir_prefix}.{self.optstep-1}',
                                     geom_path)
        geom = load_yaml(wt_input)
        layers = geom['components']['blade']['internal_structure_2d_fem']['layers']
        for name, override_dict in kwargs.items():
            for ilay,layer in enumerate(layers):
                if layer['name'] != name:
                    continue
                for key,val in override_dict.items():
                    if 'grid' in layer[key].keys():
                        print('Overriding gridded values for',key,'in layer',ilay,name)
                        assert len(val) == len(layer[key]['grid'])
                        # workaround for ruamel_yaml.representer.RepresenterError
                        layers[ilay][key]['values'] = [float(v) for v in val]
                    else:
                        print('Overriding value for',key,'in layer',ilay,name)
                        layers[ilay][key]['values'] = float(val)
        if geom_path:
            # write a new geometry file, which needs to be specified when
            # calling optimize()
            write_yaml(geom, geom_path)
            return geom_path
        else:
            # overwrite the geometry file from the previous step (we created a
            # backup)
            write_yaml(geom, wt_input)

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
        write_yaml(self.aopt, fpath_analysis_options)
        write_yaml(self.mopt, fpath_modeling_options)
        runscript = os.path.join(
                self.run_dir, f'{self.runscript_prefix}.{self.optstep}.py')
        if len(model_changes) > 0:
            for key, val in model_changes.items():
                if hasattr(val,'__iter__') and not isinstance(val, (str,)):
                    model_changes[key] = list(val)
        with open(runscript,'w') as f:
            f.write('from wisdem import run_wisdem\n')
            if len(model_changes) > 0:
                f.write('\nmodel_changes = ')
                pprint(model_changes, f)
            f.write(f"""
wt_opt, modeling_options, opt_options = run_wisdem(
    '{fpath_wt_input}',
    '{fpath_modeling_options}',
    '{fpath_analysis_options}'""")
            if len(model_changes) > 0:
                f.write(',\n\toverridden_values=model_changes')
            f.write(')\n')
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
        os.chmod(fpath, 0o755) # need to convert mode to octal
        print('\nWrote postprocessing script',fpath)


    # getter functions for convenience
    def get_tower_diameter(self):
        return self.wt_opt['towerse.tower_outer_diameter']['val']
    def get_tower_thickness(self):
        return self.wt_opt['towerse.tower_wall_thickness']['val']
    def get_tower_zpts(self):
        return self.wt_opt['towerse.z_param']['val']


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

        for layer in self.aopt['design_variables']['blade']['structure']:
            layer_name = layer['layer_name'] # e.g., Spar_Cap_*
            # handle special case of pressure-side spar-cap thickness
            # equal to the suction side -- don't need to add extra
            # control points
            equal_to_suction = layer.get('equal_to_suction',False)
            if equal_to_suction:
                print('Pressure-side spar-cap thickness '
                      'equal to suction side')
                if 'n_opt' in layer.keys():
                    print(f'Ignoring {layer_name}.n_opt =', layer['n_opt'])
                continue
            to_opt.append(f'blade:structure:{layer_name}')
            try:
                # optimize control points from index_start to index_end-1
                n_opt = layer['index_end'] - layer['index_start']
            except KeyError:
                # range not specified, optimizing all control points
                n_opt = layer['n_opt']
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
                    grid = self.get_tower_diameter()
                elif prop == 'layer_thickness':
                    grid = self.get_tower_thickness()
                elif prop == 'section_height':
                    grid = self.get_tower_zpts()
                n_opt = len(grid)
                n_fd += dv_fac * n_opt

        print('Optimizations:',to_opt)
        print('Number of finite differences needed:',n_fd)
        return n_fd


    def _execute(self, runscript=None, serial=False):
        if runscript is None:
            runscript = f'{self.runscript_prefix}.{self.optstep}.py'
        cmd = [sys.executable, runscript]

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

        
    def optimize(self,
                 label=None,
                 geom_path=None,
                 override_dict={},
                 rerun=False,
                 serial=False):
        """Run an optimization step

        Parameters
        ----------
        label: str, optional
            Name of this optimization that is copied into the
            postprocessing script
        geom_path: str, optional
            Geometry yaml file to use as a starting point; by default,
            set to the geometry file from the previous optimization step
        override_dict: dict, optional
            Dictionary of values to update in the input geometry prior
            to running the optimization
        rerun: bool, optional
            Repeat the optimization step even if output already exists
        serial: bool, optional
            Run without MPI
        """
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
            if geom_path is None:
                wt_input = os.path.join(f'{self.outdir_prefix}.{self.optstep-1}',
                                        f'{self.prefix}-step{self.optstep-1}.yaml')
            else:
                assert os.path.isfile(geom_path)
                wt_input = geom_path
            print('Starting geometry:',wt_input)
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
                    fpath_wt_input,
                    fpath_modeling_options,
                    fpath_analysis_options,
                    model_changes=override_dict)
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
                                 f'{self.prefix}-step{self.optstep}.pkl')
        if not os.path.isfile(wt_output):
            sys.exit(f'Problem with optimization step {self.optstep}, '
                     f'{wt_output} not found')

        # load new turbine data object, an instance of
        # openmdao.core.problem.Problem
        #
        # **NOTE**: This doesn't work, however, because importing load_wisdem
        # inevitably imports mpi_tools.py, which then imports mpi4py, after
        # which subprocess is unable to call mpirun...
        #self.wt_opt, _, _ = load_wisdem(wt_output)
        self.wt_opt = load_pickle(wt_output) 

        # load new turbine geometry file
        # - probably easier to work with than the pickled output
        new_geom = os.path.join(outdir,
                                f'{self.prefix}-step{self.optstep}.yaml')
        self.geom = load_yaml(new_geom)

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
        rna_mass = float(self.wt_opt['drivese.rna_mass']['val'][0])
        rna_cg = [float(val) for val in self.wt_opt['towerse.rna_cg']['val']]
        rna_I = [float(val) for val in self.wt_opt['towerse.rna_I']['val']]
        rna_F = [float(val) for val in
                 self.wt_opt['towerse.tower.rna_F']['val'].squeeze()]
        rna_M = [float(val) for val in
                 self.wt_opt['towerse.tower.rna_M']['val'].squeeze()]
        if self.mopt['WISDEM']['RotorSE']['flag']:
            Vrated = float(
                    self.wt_opt['rotorse.rp.powercurve.rated_V']['val'][0])
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
