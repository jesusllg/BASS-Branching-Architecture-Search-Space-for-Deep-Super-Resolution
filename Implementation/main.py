# main.py

import numpy as np
from nsga3 import NSGA3
from evaluation import evaluate_network_with_synflow, calculate_model_flops
from model_builder import get_model
from encoding import decode, bstr_to_rstr
import tensorflow as tf

class ModifiedMyPMOP:
    def __init__(self, n_var=84, n_obj=3):
        self.n_var = n_var
        self.n_obj = n_obj
        self.xl = np.zeros(self.n_var)  # Lower bounds
        self.xu = np.ones(self.n_var)  # Upper bounds
        self.input_size = (1, 32, 32, 3)
        self.seed_value = 1  # For reproducibility

    def func_eval_model(self, ind, n_eval):
        genotype = decode(bstr_to_rstr(ind))
        model = get_model(genotype)

        if not model.built:
            model.build(input_shape=(None,) + model.input_shape[1:])

        y_pred = evaluate_network_with_synflow(model, self.input_size)
        return y_pred

    def func_eval_flops(self, ind):
        genotype = decode(bstr_to_rstr(ind))
        model = get_model(genotype)
        flops = calculate_model_flops(model)
        return flops

    def func_eval_params(self, ind):
        genotype = decode(bstr_to_rstr(ind))
        model = get_model(genotype)
        params = model.count_params()
        return params

    def _evaluate_multi(self, ind, n_eval):
        f1 = self.func_eval_model(ind, n_eval)
        f2 = self.func_eval_params(ind)
        f3 = self.func_eval_flops(ind)
        return [-f1, f2, f3]

if __name__ == "__main__":
    problem = ModifiedMyPMOP()
    nsga3 = NSGA3(pop_size=20, n_gen=1250, problem=problem, verbose=True)
    final_population, non_dominated_solutions = nsga3()
    print("Final Population:")
    print(final_population)
    print("Non-Dominated Solutions:")
    print(non_dominated_solutions)
