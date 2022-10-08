import abc
import numpy as np

from typing import Union, Optional, Dict


class BaseMultiTaskSampler(metaclass=abc.ABCMeta):
    def __init__(self, task_dict: dict, rng: Union[int, np.random.RandomState, None]):
        self.task_dict = task_dict
        if isinstance(rng, int) or rng is None:
            rng = np.random.RandomState(rng)
        self.rng = rng

    def pop(self):
        raise NotImplementedError()

    def iter(self):
        yield self.pop()


class UniformMultiTaskSampler(BaseMultiTaskSampler):
    def pop(self):
        task_name = self.rng.choice(list(self.task_dict))
        return task_name, self.task_dict[task_name]


class ProportionalMultiTaskSampler(BaseMultiTaskSampler):
    def __init__(
        self,
        task_dict: dict,
        rng: Union[int, np.random.RandomState],
        task_to_num_examples_dict: dict,
    ):
        super().__init__(task_dict=task_dict, rng=rng)
        assert task_dict.keys() == task_to_num_examples_dict.keys()
        self.task_to_examples_dict = task_to_num_examples_dict
        self.task_names = list(task_to_num_examples_dict.keys())
        self.task_num_examples = np.array([task_to_num_examples_dict[k] for k in self.task_names])
        self.task_p = self.task_num_examples / self.task_num_examples.sum()

    def pop(self):
        task_name = self.rng.choice(self.task_names, p=self.task_p)
        return task_name, self.task_dict[task_name]


class SpecifiedProbMultiTaskSampler(BaseMultiTaskSampler):
    def __init__(
        self,
        task_dict: dict,
        rng: Union[int, np.random.RandomState],
        task_to_unweighted_probs: dict,
    ):
        super().__init__(task_dict=task_dict, rng=rng)
        assert task_dict.keys() == task_to_unweighted_probs.keys()
        self.task_to_unweighted_probs = task_to_unweighted_probs
        self.task_names = list(task_to_unweighted_probs.keys())
        self.unweighted_probs_arr = np.array([task_to_unweighted_probs[k] for k in self.task_names])
        self.task_p = self.unweighted_probs_arr / self.unweighted_probs_arr.sum()

    def pop(self):
        task_name = self.rng.choice(self.task_names, p=self.task_p)
        return task_name, self.task_dict[task_name]


class TemperatureMultiTaskSampler(BaseMultiTaskSampler):
    def __init__(
        self,
        task_dict: dict,
        rng: Union[int, np.random.RandomState],
        task_to_num_examples_dict: dict,
        temperature: float,
        examples_cap: Optional[int],
    ):
        super().__init__(task_dict=task_dict, rng=rng)
        assert task_dict.keys() == task_to_num_examples_dict.keys()
        self.task_to_num_examples_dict = task_to_num_examples_dict
        self.temperature = temperature
        self.examples_cap = examples_cap
        self.task_names = list(task_to_num_examples_dict.keys())
        self.task_num_examples = np.array([task_to_num_examples_dict[k] for k in self.task_names])
        raw_n = self.task_num_examples.clip(max=examples_cap) ** (1 / self.temperature)
        self.task_p = raw_n / raw_n.sum()

    def pop(self):
        task_name = self.rng.choice(self.task_names, p=self.task_p)
        return task_name, self.task_dict[task_name]

    def get_task_p(self):
        assert len(self.task_names) == len(self.task_p)
        return {task_name: p for task_name, p in zip(self.task_names, self.task_p)}
