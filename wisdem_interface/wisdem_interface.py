import os
import sys
import subprocess
import time

from wisdem_interface.helpers import load_yaml, save_yaml


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
                 runscript_prefix='run_wisdem',
                 mpirun='mpirun'):
        self.run_dir = run_dir
        self.prefix = turbine_prefix
        self.runscript_prefix = runscript_prefix
        self.mpirun = mpirun

        self.mopt = load_yaml(default_modeling_options)
        self.aopt = load_yaml(default_analysis_options)

        try:
            self.maxranks = int(os.environ['SLURM_NTASKS'])
        except KeyError:
            self.maxranks = 1
        print('Maximum number of MPI ranks =',self.maxranks)

        self.optstep = 0
        self.optlabels = []
        self.outfpaths = []
        self.optimize(label='Baseline',geom_path=starting_geometry)


    def __del__(self):
        self._write_postproc_script()


    def _write_inputs_and_runscript(self,
                                    fpath_wt_input,
                                    fpath_modeling_options,
                                    fpath_analysis_options,
                                    model_changes={}):
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


    def _write_postproc_script(self):
        fpath = f'compare_{self.prefix}_designs.sh'
        labels = [f'"{label}"' for label in self.optlabels]
        cmd = 'compare_designs ' \
            + ' '.join(self.outfpaths) \
            + ' --labels ' \
            + ' '.join(labels)
        with open(fpath,'w') as f:
            f.write('grep "^Optimization" log.wisdem.?\n')
            f.write(cmd)
        print('\nWrote postprocessing script',fpath)


    def _get_num_finite_differences(self):
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
                n_fd += dv_fac
        print('Optimizations:',to_opt)
        print('Number of finite differences needed:',n_fd)
        return n_fd


    def _run(self, runscript=None):
        n_fd = self._get_num_finite_differences()
        nranks = min(max(1,n_fd), self.maxranks)

        if runscript is None:
            runscript = f'{self.runscript_prefix}.{self.optstep}.py'
        runscript = ['python',runscript]

        if n_fd > 0:
            runscript = [self.mpirun,'-n',str(nranks)] + runscript

        print('Executing:',' '.join(runscript))
        with open(f'log.wisdem.{self.optstep}','w') as log:
            subprocess.run(
                    runscript, stdout=log, stderr=subprocess.STDOUT, text=True)

        
    def optimize(self, label=None, geom_path=None, rerun=False):
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
            print('\n=== Running optimization step',self.optstep,'===')
            wt_input = os.path.join(f'outputs.{self.optstep-1}',
                                    f'{self.prefix}-step{self.optstep-1}.yaml')
            wt_output = f'{self.prefix}-step{self.optstep}.yaml'

        # create new output dir
        outdir = os.path.join(self.run_dir, f'outputs.{self.optstep}')
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
            self._run(runscript)
            print('Run time: %f'%(time.time()-tt))
            sys.stdout.flush()
        else:
            print(full_wt_output_path,'found,'
                  ' set rerun=True to repeat this optimization step')

        self.optstep += 1

