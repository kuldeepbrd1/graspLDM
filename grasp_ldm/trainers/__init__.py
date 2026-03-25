import enum

from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger

LOGGERS = {
    "WandbLogger": WandbLogger,
    "TensorBoardLogger": TensorBoardLogger,
    "CSVLogger": CSVLogger,
}


class E_Trainers(enum.Enum):
    VAE = "vae"
    DDM = "ddm"

    def __repr__(self):
        return f"{self.__class__.__name__}.{self.name}"

    @staticmethod
    def _get_trainer(model_type):
        if model_type == E_Trainers.VAE:
            from grasp_ldm.trainers.grasp_generation_trainer import GraspVAETrainer
            return GraspVAETrainer
        elif model_type == E_Trainers.DDM:
            from grasp_ldm.trainers.grasp_generation_trainer import GraspLDMTrainer
            return GraspLDMTrainer
        raise NotImplementedError(f"Model type {model_type} not implemented")

    def get_trainer(self):
        return E_Trainers._get_trainer(self)

    @staticmethod
    def from_string(model_type: str):
        try:
            return E_Trainers(model_type)
        except ValueError:
            raise NotImplementedError(f"Model type '{model_type}' not implemented. Choose from: {[e.value for e in E_Trainers]}")

    @staticmethod
    def get(model_type: str):
        return E_Trainers._get_trainer(E_Trainers.from_string(model_type))
