__all__ = [
    "cache_generator_call",
]

import types
from typing import Protocol, Union

import functools
import inspect
from collections import UserList
from dataclasses import dataclass, make_dataclass

from miv.core.datatype import DataTypes, Extendable
from miv.core.operator.cachable import (
    DataclassCacher,
    FunctionalCacher,
    _CacherProtocol,
)
from miv.core.operator.operator import Operator, OperatorMixin


def cache_generator_call(func):
    """
    Cache the methods of the operator.
    It is special case for the generator in-out stream.
    Save the cache in the cacher object with appropriate tag.

    If inputs are not all generators, it will run regular function.
    """

    def wrapper(self: Operator, *args, **kwargs):
        is_all_generator = all(inspect.isgenerator(v) for v in args) and all(
            inspect.isgenerator(v) for v in kwargs.values()
        )

        tag = "data"
        cacher: DataclassCacher = self.cacher

        if is_all_generator:

            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()

            def generator_func(*args):
                print(f"Hello from rank {rank} of {size} total ranks.")
                for idx, zip_arg in enumerate(zip(*args)):

                    if rank == idx % size:
                        print(f"idx {idx} handled by rank {rank}")
                        result = func(self, *zip_arg, **kwargs)
                        if result is not None:
                            # In case the module does not return anything
                            cacher.save_cache(result, idx, tag=tag)
                        if not self.skip_plot:
                            self.generator_plot(idx, result, zip_arg, save_path=True)
                            if idx == 0:
                                self.firstiter_plot(result, zip_arg, save_path=True)
                        print(f"idx {idx} finished by rank {rank}")
                        yield result
                else:
                    cacher.save_config(tag=tag)
                    # TODO: add lastiter_plot

            generator = generator_func(*args, *kwargs.values())
            return generator
        else:
            result = func(self, *args, **kwargs)
            if result is None:
                # In case the module does not return anything
                return None
            cacher.save_cache(result, tag=tag)
            cacher.save_config(tag=tag)
            return result

    return wrapper
