# nsga3.py

import numpy as np
import copy
import math
from evaluation import evaluate_model, calculate_model_flops, count_params
from utils import Dominance
from tqdm import tqdm

class ReferencePoint:
    """
    Class representing a reference point for NSGA-III.
    """
    def __init__(self, position):
        self.position = position

class NSGA3:
    """
    NSGA-III algorithm implementation.
    """
    def __init__(self, problem, pop_size=100, n_gen=100, verbose=False):
        """
        Initialize the NSGA-III optimizer.

        Args:
            problem: The optimization problem to be solved.
            pop_size: Population size.
            n_gen: Number of generations.
            verbose: If True, print verbose output.
        """
        self.problem = problem
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.verbose = verbose
        self.n_eval = 0  # Initialize evaluation counter

    def _dominate(self, p, q):
        """
        Check if one solution dominates another.

        Args:
            p: The first solution's objectives.
            q: The second solution's objectives.

        Returns:
            True if p dominates q, False otherwise.
        """
        better_in_all = all(p_i <= q_i for p_i, q_i in zip(p, q))
        better_in_any = any(p_i < q_i for p_i, q_i in zip(p, q))
        return better_in_all and better_in_any

    def _fast_non_dominated_sorting(self, pop):
        """
        Perform fast non-dominated sorting on the population.

        Args:
            pop: Population to be sorted.

        Returns:
            Sorted fronts of the population.
        """
        fronts = {}
        S = [[] for _ in range(len(pop['F']))]
        n = [0] * len(pop['F'])
        rank = [0] * len(pop['F'])
        fronts[1] = []

        for p in range(len(pop['F'])):
            S[p] = []
            n[p] = 0
            for q in range(len(pop['F'])):
                if self._dominate(pop['F'][p], pop['F'][q]):
                    S[p].append(q)
                elif self._dominate(pop['F'][q], pop['F'][p]):
                    n[p] += 1
            if n[p] == 0:
                rank[p] = 1
                fronts[1].append(p)

        i = 1
        while fronts[i]:
            Q = []
            for p in fronts[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        rank[q] = i + 1
                        Q.append(q)
            i += 1
            fronts[i] = Q

        return fronts

    def _initialize_pop(self):
        """
        Initialize the population.

        Returns:
            Initialized population.
        """
        X = []
        F = []
        for _ in range(self.pop_size):
            ind = np.random.randint(0, 2, size=self.problem.n_var)
            X.append(ind)
            obj = self.problem.evaluate(ind, self.n_eval)
            F.append(obj)
            self.n_eval += 1
        return {'X': X, 'F': F}

    def _tournament_selection(self, pop, n_parents=2):
        """
        Perform tournament selection.

        Args:
            pop: Current population.

        Returns:
            Selected parents.
        """
        selected = []
        for _ in range(n_parents):
            i, j = np.random.choice(len(pop['X']), 2, replace=False)
            if self._dominate(pop['F'][i], pop['F'][j]):
                selected.append(i)
            else:
                selected.append(j)
        return selected

    def _crossover(self, parent1, parent2, prob=0.9):
        """
        Perform k-point crossover.

        Args:
            parent1: First parent's genotype.
            parent2: Second parent's genotype.
            prob: Crossover probability.

        Returns:
            Offspring genotype.
        """
        if np.random.rand() < prob:
            k = np.random.randint(1, self.problem.n_var)
            crossover_points = np.random.choice(range(1, self.problem.n_var), k, replace=False)
            crossover_points.sort()
            offspring = parent1.copy()
            start = 0
            for i, point in enumerate(crossover_points):
                if i % 2 == 0:
                    offspring[start:point] = parent2[start:point]
                start = point
            if len(crossover_points) % 2 == 0:
                offspring[start:] = parent2[start:]
            return offspring
        else:
            return parent1.copy()

    def _mutation(self, offspring, prob=1 / 84):  # Adjusted probability
        """
        Perform bit-flip mutation.

        Args:
            offspring: Offspring genotype.
            prob: Mutation probability.

        Returns:
            Mutated offspring genotype.
        """
        for i in range(self.problem.n_var):
            if np.random.rand() < prob:
                offspring[i] = 1 - offspring[i]
        return offspring

    def _generate_offspring(self, pop):
        """
        Generate offspring population.

        Args:
            pop: Current population.

        Returns:
            Offspring population.
        """
        offspring_X = []
        offspring_F = []
        for _ in range(self.pop_size):
            parents_indices = self._tournament_selection(pop)
            parent1 = pop['X'][parents_indices[0]]
            parent2 = pop['X'][parents_indices[1]]
            offspring = self._crossover(parent1, parent2)
            offspring = self._mutation(offspring)
            obj = self.problem.evaluate(offspring, self.n_eval)
            self.n_eval += 1
            offspring_X.append(offspring)
            offspring_F.append(obj)
        return {'X': offspring_X, 'F': offspring_F}

    def generate_reference_points(self, num_objs, divisions=4):
        """
        Generate reference points for NSGA-III.

        Args:
            num_objs: Number of objectives.
            divisions: Number of divisions per objective.

        Returns:
            List of reference points.
        """
        def recursive_ref(point, left, total, depth):
            if depth == num_objs - 1:
                point[depth] = left / total
                refs.append(ReferencePoint(np.array(point)))
            else:
                for i in range(left + 1):
                    point[depth] = i / total
                    recursive_ref(point.copy(), left - i, total, depth + 1)

        refs = []
        recursive_ref([0] * num_objs, divisions, divisions, 0)
        return refs

    def perpendicular_distance(self, direction, point):
        """
        Calculate perpendicular distance from a point to a reference line.

        Args:
            direction: Reference direction.
            point: Point coordinates.

        Returns:
            Perpendicular distance.
        """
        norm_direction = np.linalg.norm(direction)
        if norm_direction == 0:
            return np.linalg.norm(point)
        scalar_proj = np.dot(point, direction) / norm_direction
        proj = scalar_proj * (direction / norm_direction)
        return np.linalg.norm(point - proj)

    def _associate(self, pop_F, fronts):
        """
        Associate solutions with reference points.

        Args:
            pop_F: Objective values of the population.
            fronts: Non-dominated fronts.

        Returns:
            Association of solutions to reference points.
        """
        associations = {}
        niche_counts = {i: 0 for i in range(len(self.ref_points))}
        for front in fronts.values():
            for idx in front:
                distances = []
                for i, ref_point in enumerate(self.ref_points):
                    dist = self.perpendicular_distance(ref_point.position, pop_F[idx])
                    distances.append((dist, i))
                min_dist, min_idx = min(distances)
                associations[idx] = (min_idx, min_dist)
                niche_counts[min_idx] += 1
        return associations, niche_counts

    def _niching(self, pop, pop_F, associations, niche_counts):
        """
        Niching process to select individuals for the next generation.

        Args:
            pop: Combined population.
            pop_F: Objective values of the combined population.
            associations: Associations of solutions to reference points.
            niche_counts: Counts of solutions associated with each reference point.

        Returns:
            Selected indices for the next generation.
        """
        next_gen_indices = []
        while len(next_gen_indices) < self.pop_size:
            min_count = min(niche_counts.values())
            min_refs = [ref for ref, count in niche_counts.items() if count == min_count]
            for ref in min_refs:
                candidates = [idx for idx, assoc in associations.items() if assoc[0] == ref and idx not in next_gen_indices]
                if candidates:
                    # Select the candidate with the smallest perpendicular distance
                    candidates.sort(key=lambda idx: associations[idx][1])
                    selected = candidates[0]
                    next_gen_indices.append(selected)
                    niche_counts[ref] += 1
                    break
            else:
                break  # No candidates left
        return next_gen_indices

    def _normalize_objectives(self, pop_F):
        """
        Normalize the objective values.

        Args:
            pop_F: Objective values of the population.

        Returns:
            Normalized objective values.
        """
        pop_array = np.array(pop_F)
        f_min = pop_array.min(axis=0)
        f_max = pop_array.max(axis=0)
        normalized = (pop_array - f_min) / (f_max - f_min + 1e-6)
        return normalized

    def _execute(self):
        """
        Execute the NSGA-III algorithm.

        Returns:
            Final population and non-dominated solutions.
        """
        self.ref_points = self.generate_reference_points(num_objs=self.problem.n_obj)
        pop = self._initialize_pop()

        if self.verbose:
            pbar = tqdm(total=self.n_gen, desc='NSGA-III Progress')

        for gen in range(self.n_gen):
            # Generate offspring
            offspring = self._generate_offspring(pop)

            # Combine parent and offspring populations
            combined_pop = {'X': pop['X'] + offspring['X'], 'F': pop['F'] + offspring['F']}

            # Perform non-dominated sorting
            fronts = self._fast_non_dominated_sorting(combined_pop)

            # Normalize objectives
            combined_F = combined_pop['F']
            normalized_F = self._normalize_objectives(combined_F)

            # Associate solutions with reference points
            associations, niche_counts = self._associate(normalized_F, fronts)

            # Niching process to select next generation
            next_gen_indices = []
            front_num = 1
            while len(next_gen_indices) + len(fronts[front_num]) <= self.pop_size:
                next_gen_indices.extend(fronts[front_num])
                front_num += 1

            if len(next_gen_indices) < self.pop_size:
                remaining = self.pop_size - len(next_gen_indices)
                last_front = fronts[front_num]
                # Recalculate associations and niche counts for the last front
                last_front_associations = {idx: associations[idx] for idx in last_front}
                selected_indices = self._niching(combined_pop, normalized_F, last_front_associations, niche_counts)
                next_gen_indices.extend(selected_indices[:remaining])

            # Update population
            pop['X'] = [combined_pop['X'][i] for i in next_gen_indices]
            pop['F'] = [combined_pop['F'][i] for i in next_gen_indices]

            if self.verbose:
                pbar.update(1)

        if self.verbose:
            pbar.close()

        # Extract non-dominated solutions
        final_fronts = self._fast_non_dominated_sorting(pop)
        nds_indices = final_fronts[1]
        nds = {'X': [pop['X'][i] for i in nds_indices], 'F': [pop['F'][i] for i in nds_indices]}

        return pop, nds

    def run(self):
        """
        Run the NSGA-III algorithm.

        Returns:
            Final population and non-dominated solutions.
        """
        return self._execute()
