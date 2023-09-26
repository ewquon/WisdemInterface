#!/usr/bin/env python
"""
Check optimality and constraint convergences (can check while running)

This script expects that output folders are organized by WisdemInterface:

    design.0
    ├─ prefix-step0-analysis.yaml
    ├─ prefix-step0-modeling.yaml
    └─ ...
    design.1
    ├─ prefix-step1-analysis.yaml
    ├─ prefix-step1-modeling.yaml
    └─ ...
    ...
    design.N
    run_wisdem.0.py
    run_wisdem.1.py
    ...
    run_wisdem.N.py

This structure enables `compare_designs` to work properly but trips up the
PlotRecorder here.

Usage
-----
check_convergence.py outputs.N
"""
import sys
import os
import glob
import openmdao.api as om
import wisdem.inputs as sch
from wisdem.glue_code.gc_RunTools import PlotRecorder

assert os.path.isdir(sys.argv[1])
flist = glob.glob(os.path.join(sys.argv[1],'*-analysis.yaml'))
assert len(flist) == 1
fname_analysis_options = flist[0]

print('Loading',fname_analysis_options)
analysis_options = sch.load_analysis_yaml(fname_analysis_options)

# workaround for analysis file being inside output dir
analysis_options['general']['output_dir'] = '.'

# output convergence trends
wt_opt = om.Problem(model=PlotRecorder(opt_options=analysis_options), reports=False)
wt_opt.setup(derivatives=False)
wt_opt.run_model()
