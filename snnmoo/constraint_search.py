from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.problem import Problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter


from snnmoo.snn import SNN, SNNFirings

def snn_firing_test(
    ex_target_arg: float, 
    in_target_arg: float, 
    sparsity: float, 
    max_error: float,
    generations: int=100, 
    pop_size: int=100
    ) -> Tuple[Any, Dict]:
    ''' Create a new problem class and run to get results '''
    
    class SNNProblem(Problem):

        def __init__(self) -> None:
            super().__init__(n_var=2,
                             n_obj=2,
                             n_constr=2,
                             xl=np.array([0, 0]),
                             xu=np.array([2, 2]),
                             elementwise_evaluation=True)

        def _evaluate(self, chromosome: np.array, out: Dict, *args: List, **kwargs:  Dict) -> None:
            model = SNN(
                ge = chromosome[0],
                gi = chromosome[1],
                sparsity = sparsity,
            )

            firing = model.run_network()

            ex_target = ex_target_arg
            ex_distance = abs(ex_target - firing.score()['ex_firing'])

            in_target = in_target_arg
            in_distance = abs(in_target - firing.score()['in_firing'])
            
            # objective
            out["F"] = [ex_distance, in_distance]
            # constraint
            out["G"] = [ex_distance-max_error, in_distance-max_error]


    problem = SNNProblem()

    algorithm = NSGA2(pop_size=pop_size)

    res = minimize(problem,
                   algorithm,
                   ("n_gen", generations),
                   verbose=True,
                   save_history=True,
                   seed=1)

    params = {
        'ex_target':   ex_target_arg,
        'in_target':   in_target_arg,
        'sparsity' :   sparsity,
        'max_error':   max_error,
        'generations': generations,
        'pop_size':    pop_size,
    }

    return res, params

import os
import json

def serialse_results(res: Any, params: Dict, output: str) -> None:
    os.mkdir(output)
    n_evals = np.array([e.evaluator.n_eval for e in res.history])
    opt = np.array([e.opt[0].F for e in res.history])
    
    x=np.vstack([n_evals,opt[:,0],opt[:,1]]).T

    d = pd.DataFrame(x, columns=['n_evals', 'ex', 'inhib'])
    d.to_csv(f'{output}/opt.csv')
    
    frontier = pd.DataFrame(res.F, columns=['ex_firing_error','inhib_firing_error'])
    frontier.to_csv(f'{output}/frontier.csv')

    with open(f'{output}/params.json', 'w') as f:
        json.dump( params, f )


firing_rates = [
    (2,10),
    (10,2),
    (2,2)
]

sparsities = [
    1, .5, .2
]

import sys
import datetime

max_error = 5

for sparsity in sparsities:
    for ex, inh in firing_rates:
        res, params = snn_firing_test(ex, inh, sparsity, max_error, pop_size=50, generations=25)
        time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        serialse_results(res, params, f'/users/jfitzgerald/results/constraint/{time_str}')

