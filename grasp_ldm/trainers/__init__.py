import enum

from pytorch_lightning.loggers import CSVLogger, Logger, TensorBoardLogger, WandbLogger

LOGGERS = {
    "WandbLogger": WandbLogger,
    "TensorBoardLogger": TensorBoardLogger,
    "CSVLogger": CSVLogger,
}


class E_Trainers(enum.Enum):
    CLASSIFIER = "classifier"
    VAE = "vae"
    DDM = "ddm"

    def __repr__(self):
        return f"{self.__class__.__name__}.{self.name}"

    def _get_trainer(model_type: str):
        if model_type == E_Trainers.CLASSIFIER:
            from grasp_ldm.trainers.grasp_classification_trainer import (
                GraspClassificationTrainer,
            )

            return GraspClassificationTrainer
        elif model_type == E_Trainers.VAE:
            from grasp_ldm.trainers.grasp_generation_trainer import GraspVAETrainer

            return GraspVAETrainer
        elif model_type == E_Trainers.DDM:
            from grasp_ldm.trainers.grasp_generation_trainer import GraspLDMTrainer

            return GraspLDMTrainer
        else:
            raise NotImplementedError(f"Model type {model_type} not implemented")

    def get_trainer(self):
        return E_Trainers._get_trainer(self)

    def from_string(model_type: str):
        if model_type == "classifier":
            return E_Trainers.CLASSIFIER
        elif model_type == "vae":
            return E_Trainers.VAE
        elif model_type == "ddm":
            return E_Trainers.DDM
        else:
            raise NotImplementedError(f"Model type {model_type} not implemented")

    def get(model_type: str):
        enum_type = E_Trainers.from_string(model_type)
        return E_Trainers._get_trainer(enum_type)
