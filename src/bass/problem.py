"""Compatibility constructor dispatching to a version-specific problem."""


class BASSProblem:
    def __new__(cls, *, genome_version: int = 1, **kwargs):
        if genome_version == 1:
            from .v1.problem import BASSProblem as V1Problem

            return V1Problem(**kwargs)
        if genome_version == 2:
            from .v2.problem import BASSProblem as V2Problem

            return V2Problem(**kwargs)
        if genome_version == 3:
            from .v3.problem import BASSProblem as V3Problem

            return V3Problem(**kwargs)
        raise ValueError("genome_version must be 1, 2, or 3")
