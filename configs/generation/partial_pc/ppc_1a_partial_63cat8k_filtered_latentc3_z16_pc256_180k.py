import os

## --------------------  Most frequently changed params here  --------------------

resume_training_from_last = True

max_steps = 180000
batch_size = 60

num_gpus = 1
num_workers_per_gpu = 7

# During training, if a ckpt is provided here, it overrides resume_training_from_last and instead resumes training from this ckpt
vae_ckpt_path = None  # "output/boilerplate_kldanneal_c0.1/vae/checkpoints/last.ckpt"
ddm_ckpt_path = None

max_scenes = None


root_data_dir = "data/acronym/renders/objects_filtered_grasps_63cat_8k/"
camera_json = "grasp_ldm/dataset/cameras/camera_d435i_dummy.json"

## -------------------- Inputs/Shapes ------------------------
# Input/Output: grasp representation [mrp(3), t(3), cls_success(1), qualities(4)]

pc_num_points = 1024
pc_latent_dims = 256
pc_latent_channels = 3

grasp_pose_dims = 6
num_output_qualities = 0
grasp_latent_dims = 16

grasp_representation_dims = (
    grasp_pose_dims + num_output_qualities + 1
    if num_output_qualities is not None
    else grasp_pose_dims + 1
)

## ----------------------- Model -----------------------

dropout = 0.1  # or None

pc_encoder_config = dict(
    type="PVCNNEncoder",
    args=dict(
        in_features=3,
        n_points=pc_num_points,
        scale_channels=0.75,
        scale_voxel_resolution=0.75,
        num_blocks=(1, 1, 1, 1),
        out_channels=pc_latent_channels,
        use_global_attention=False,
    ),
)

grasp_encoder_config = dict(
    type="ResNet1D",
    args=dict(
        in_features=grasp_representation_dims,
        block_channels=(32, 64, 128, 256),
        input_conditioning_dims=pc_latent_dims,
        resnet_block_groups=4,
        dropout=dropout,
    ),
)

decoder_config = dict(
    type="ResNet1D",
    args=dict(
        block_channels=(32, 64, 128, 256),
        # out_dim=grasp_pose_dims,
        input_conditioning_dims=pc_latent_dims,
        resnet_block_groups=4,
        dropout=dropout,
    ),
)

loss_config = dict(
    reconstruction_loss=dict(
        type="GraspReconstructionLoss",
        name="reconstruction_loss",
        args=dict(translation_weight=1, rotation_weight=1),
    ),
    latent_loss=dict(
        type="VAELatentLoss",
        args=dict(
            name="grasp_latent",
            cyclical_annealing=True,
            num_steps=max_steps,
            num_cycles=1,
            ratio=0.5,
            start=1e-7,
            stop=0.1,
        ),
    ),
    classification_loss=dict(type="ClassificationLoss", args=dict(weight=0.1)),
    # quality_loss=dict(type="QualityLoss", args=dict(weight=0.1)),
)

denoiser_model = dict(
    type="TimeConditionedResNet1D",
    args=dict(
        dim=grasp_latent_dims,
        channels=1,
        block_channels=(32, 64, 128, 256),
        input_conditioning_dims=pc_latent_dims,
        resnet_block_groups=4,
        dropout=dropout,
        is_time_conditioned=True,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=True,
        # learned_sinusoidal_dim=16,
    ),
)
# Use `model` for single module to be built. If a list of modules are required to be built, use `models` to make sure the outer
# See models/builder.py for more info.
model = dict(
    vae=dict(
        model=dict(
            type="GraspCVAE",
            args=dict(
                grasp_latent_size=grasp_latent_dims,
                pc_latent_size=pc_latent_dims,
                pc_encoder_config=pc_encoder_config,
                grasp_encoder_config=grasp_encoder_config,
                decoder_config=decoder_config,
                loss_config=loss_config,
                num_output_qualities=num_output_qualities,
                intermediate_feature_resolution=16,
            ),
        ),
        ckpt_path=vae_ckpt_path,
    ),
    ddm=dict(
        model=dict(
            type="GraspLatentDDM",
            args=dict(
                model=denoiser_model,
                latent_in_features=grasp_latent_dims,
                diffusion_timesteps=1000,
                noise_scheduler_type="ddpm",
                diffusion_loss="l2",
                beta_schedule="linear",
                is_conditioned=True,
                joint_training=False,
                denoising_loss_weight=1,
                variance_type="fixed_large",
                elucidated_diffusion=False,
                beta_start=0.00005,
                beta_end=0.001,
            ),
        ),
        ckpt_path=ddm_ckpt_path,
        use_vae_ema_model=True,
    ),
)
## -- Data --
augs_config = [
    dict(type="RandomRotation", args=dict(p=0.5, max_angle=180, is_degree=True)),
    dict(type="PointcloudJitter", args=dict(p=1, sigma=0.005, clip=0.005)),
    dict(type="RandomPointcloudDropout", args=dict(p=0.5, max_dropout_ratio=0.4)),
]


train_data = dict(
    type="AcronymPartialPointclouds",
    args=dict(
        data_root_dir=root_data_dir,
        max_scenes=max_scenes,
        camera_json=camera_json,
        num_points_per_pc=pc_num_points,
        num_grasps_per_obj=100,
        rotation_repr="mrp",
        augs_config=augs_config,
        split="train",
        depth_px_scale=10000,
        scene_prefix="scene_",
        min_usable_pc_points=1024,
        preempt_load_data=True,
        use_failed_grasps=False,
        failed_grasp_ratio=0.3,
        load_fixed_grasp_transforms=None,
        is_input_dataset_normalized=False,
        num_repeat_dataset=10,
    ),
    batch_size=batch_size,
)

data = dict(
    train=train_data,
)

# Patch: Mesh Categories. Used for simulation
mesh_root = root_data_dir
mesh_categories = [
    "Cup",
    "Mug",
    "Fork",
    "Hat",
    "Bottle",
    "Bowl",
    "Car",
    "Donut",
    "Laptop",
    "MousePad",
    "Pencil",
    "Plate",
    "ScrewDriver",
    "WineBottle",
    "Backpack",
    "Bag",
    "Banana",
    "Battery",
    "BeanBag",
    "Bear",
    "Book",
    "Books",
    "Camera",
    "CerealBox",
    "Cookie",
    "Hammer",
    "Hanger",
    "Knife",
    "MilkCarton",
    "Painting",
    "PillBottle",
    "Plant",
    "PowerSocket",
    "PowerStrip",
    "PS3",
    "PSP",
    "Ring",
    "Scissors",
    "Shampoo",
    "Shoes",
    "Sheep",
    "Shower",
    "Sink",
    "SoapBottle",
    "SodaCan",
    "Spoon",
    "Statue",
    "Teacup",
    "Teapot",
    "ToiletPaper",
    "ToyFigure",
    "Wallet",
    "WineGlass",
    "Cow",
    "Sheep",
    "Cat",
    "Dog",
    "Pizza",
    "Elephant",
    "Donkey",
    "RubiksCube",
    "Tank",
    "Truck",
    "USBStick",
]

## --------------------  Trainer  --------------------
## Logger
logger = dict(type="WandbLogger", project="partial-pc-63c-ema")

optimizer = dict(
    initial_lr=0.001,
    scheduler=dict(
        type="MultiStepLR",
        args=dict(milestones=[int(max_steps / 3), int(2 * max_steps / 3)], gamma=0.1),
    ),
)

trainer = dict(
    max_steps=max_steps,
    batch_size=batch_size,
    num_workers=num_workers_per_gpu * num_gpus,
    accelerator="gpu",
    devices=num_gpus,
    strategy="ddp",
    logger=logger,
    log_every_n_steps=100,
    optimizer=optimizer,
    resume_training_from_last=resume_training_from_last,
    check_val_every_n_epoch=1,
    ema=dict(
        beta=0.990,
        update_after_step=1000,
    ),
    deterministic=True,
)
