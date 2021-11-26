from psfMC import model_galaxy_mcmc
import os
import glob
from multiprocessing import Pool
_num_procs = 16

# These are additional parameters for the MCMC fitter. Number of iterations,
# number of burn-in iterations (which are discarded)
_mcparams = {'burn': 800, 'iterations': 800, 'chains': 50}
_extensions = ['MMBH','SDSS','CO','WFIRST']

def run_mcmc(model_file,ext):
    output_name = model_file.replace('model', 'out').replace('.py', '')
    if len(glob.glob(output_name+'*fits*')) > 0:
        print('{} already processed, skipping'.format(model_file))
        return
    model_galaxy_mcmc(model_file, output_name=output_name, **_mcparams)


if __name__ == '__main__':
    for ext in _extensions:
      model_files = glob.glob('mcmc_model_mock_JWST_{}.py'.format(ext))
      print(model_files)        
      pool = Pool(_num_procs)

      results = [pool.apply_async(run_mcmc, args=(model,ext))
          for model in model_files]
      results = [result.get() for result in results]

      pool.close()
      pool.join()
    print('All done!')

