from abc import abstractmethod
from typing import Optional

from torch import Tensor, nn


class BaseGraspSampler(nn.Module):
    """Base abstract class for Grasp Samplers"""

    def __init__(self):
        super(BaseGraspSampler, self).__init__()

    @property
    def _type(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def generate_grasps(
        self, z: Optional[Tensor] = None, z_cond: Optional[Tensor] = None
    ) -> Tensor:
        """Abstract method for generating grasp poses given latents (optional: None)
        and conditioning input z_cond
        """
        raise NotImplementedError


class BaseGraspClassifier(nn.Module):
    """Base abstract class for Grasp Samplers"""

    def __init__(self):
        super(BaseGraspClassifier, self).__init__()

    @property
    def _type(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def classify_grasps(
        self, grasp_poses: Optional[Tensor] = None, pc: Optional[Tensor] = None
    ) -> Tensor:
        """Abstract method for generating grasp poses given latents (optional: None)
        and conditioning input z_cond
        """
        raise NotImplementedError
