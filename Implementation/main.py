# main.py

import numpy as np
from encoding import decode
from evaluation import evaluate_model
from nsga3 import NSGA3
from config import NSGA3_CONFIG

class OptimizationProblem:
    """
    The problem class that includes evaluation functions for the optimization problem.
    """

    def __init__(self, n_var=84, n_obj=3):
        self.n_var = n_var
        self.n_obj = n_obj
        self.xl = np.zeros(self.n_var)  # Lower bounds
        self.xu = np.ones(self.n_var)   # Upper bounds

    def evaluate(self, ind):
        """
        Evaluate the individual and return the objective values.

        Args:
            ind (np.ndarray): Individual genotype (bitstring).

        Returns:
            list: List of objective values [evaluation_metric, params, flops].
        """
        # Decode the individual (bitstring to genotype)
        genotype = decode([ind])

        # Objective 1: Evaluation metric (negative PSNR or SynFlow)
        f1 = evaluate_model(genotype)

        # Objective 2: Number of parameters
        f2 = self.func_eval_params(genotype)

        # Objective 3: FLOPs
        f3 = self.func_eval_flops(genotype)

        return [f1, f2, f3]

    def func_eval_params(self, genotype):
        """
        Calculate the number of parameters of the model.

        Args:
            genotype: The genotype representing the model architecture.

        Returns:
            int: Number of parameters.
        """
        from model_builder import get_model
        model = get_model(genotype)
        params = model.count_params()
        tf.keras.backend.clear_session()
        return params

    def func_eval_flops(self, genotype):
        """
        Calculate the FLOPs of the model.

        Args:
            genotype: The genotype representing the model architecture.

        Returns:
            int: FLOPs count.
        """
        from model_builder import get_model
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

        model = get_model(genotype)
        if not model.built:
            model.build(input_shape=(None, 64, 64, 3))

        @tf.function
        def forward_pass(inputs):
            return model(inputs)

        concrete_func = forward_pass.get_concrete_function(
            tf.TensorSpec(shape=(1, 64, 64, 3), dtype=tf.float32)
        )

        frozen_func, _ = convert_variables_to_constants_v2_as_graph(concrete_func)
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(
            graph=frozen_func.graph, run_meta=run_meta, cmd='op', options=opts
        )
        total_flops = flops.total_float_ops
        tf.keras.backend.clear_session()
        return total_flops

if __name__ == "__main__":
    # Initialize the problem
    problem = OptimizationProblem()

    # NSGA-III parameters from config
    pop_size = NSGA3_CONFIG['POP_SIZE']
    n_gen = NSGA3_CONFIG['N_GEN']

    # Initialize NSGA-III optimizer
    optimizer = NSGA3(problem, pop_size=pop_size, n_gen=n_gen, verbose=True)

    # Run the optimization
    pop, nds = optimizer.run()

    # Print the non-dominated solutions
    print("Non-dominated solutions:")
    for ind, obj in zip(nds['X'], nds['F']):
        print(f"Solution: {ind}, Objectives: {obj}")
