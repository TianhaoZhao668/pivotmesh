from trainer import MeshAutoencoderTrainer, trackers
from model.meshAE import MeshAutoencoder
from data.dataset import ShapeNetCore, Objaverse, ShapeNetCore_obj
from accelerate.utils import DistributedDataParallelKwargs
import os
import yaml


with open("configs/AE.yaml","r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

autoencoder = MeshAutoencoder(
    num_discrete_coors = config['num_discrete_coors'],
    encoder_dim = config['encoder_dim'],
    encoder_depth = config['encoder_depth'],
    patch_size = config['patch_size'],
    dropout = config['dropout'],
    bin_smooth_blur_sigma = config['bin_smooth_blur_sigma'],
    quant_face = config['quant_face'],
    decoder_fine_dim = config['decoder_fine_dim'],
    graph_layers = config['graph_layers'],

    latent_dim = config['latent_dim'],
    pos_embed = config['pos_embed'],

    decoder_depth = config['decoder_depth'],
    decoder_dim = config['decoder_dim'],
)

if config['resume']:
    autoencoder.load(config['resume'])

if config['dataset_name'] == 'shapenet':
    TRAIN_PATH = f"/root/shapenet_data/ShapeNetCore_decimates_800face_train"
    # TRAIN_PATH = f"/root/mesh_compress/data/train/train_mesh_data_under_25kb_100"
    VAL_PATH = f"/root/shapenet_data/ShapeNetCore_decimates_800face_test"
    # VAL_PATH = f"/root/mesh_compress/data/train/train_mesh_data_under_25kb_100"
    train_dataset = ShapeNetCore_obj(
        TRAIN_PATH,
        return_pivot=False,
        augment=True,
        augment_dict=config['augment_dict'],
        quant_bit=config['quant_bit']
    )

    val_dataset = ShapeNetCore_obj(
        VAL_PATH,
        return_pivot=False,
        augment=True,
        augment_dict=config['augment_dict'],
        quant_bit=config['quant_bit']
    )
    
# elif config['dataset_name'] in ['objaverse', 'objaversexl']:
#     TRAIN_PATH = f"{os.environ['HOME']}/data/objaverse-lp-500/train-500"
#     VAL_PATH = f"{os.environ['HOME']}/data/objaverse-lp-500/val-500"
#     train_dataset = Objaverse(TRAIN_PATH, return_pivot=False, augment=True, 
#                               augment_dict=config['augment_dict'], quant_bit=config['quant_bit'])
#     val_dataset = Objaverse(VAL_PATH, return_pivot=False, augment=True, 
#                             augment_dict=config['augment_dict'], quant_bit=config['quant_bit'])

else:
    raise NotImplementedError

config['train_path'] = TRAIN_PATH
config['val_path'] = VAL_PATH

trainer = MeshAutoencoderTrainer(
    model = autoencoder,
    dataset = train_dataset,
    num_train_steps = int(config['num_train_steps']),
    val_dataset = val_dataset,
    val_every = config['val_every'],
    val_num_batches = 10,
    learning_rate = config['learning_rate'],
    batch_size = config['batch_size'],
    grad_accum_every = config['grad_accum_every'],
    warmup_steps = config['warmup_steps'], 
    weight_decay = config['weight_decay'],
    max_grad_norm = config['max_grad_norm'],
    use_wandb_tracking = False,
    checkpoint_every = config['checkpoint_every'],
    checkpoint_folder = f'checkpoints/{config["exp_name"]}',
    accelerator_kwargs = {
        'kwargs_handlers': [
            DistributedDataParallelKwargs(find_unused_parameters=False)
        ]
    }
)

with trackers(trainer, project_name='PivotMesh', run_name=config['exp_name'], hps=config):
    trainer.load('/root/github_code/pivotmesh/checkpoints/AE-shapenet_14400*10_8channel_12en_768_18de_384_1e-4kl_1GCN_womask_ordered/mesh-autoencoder.ckpt.36.pt')
    trainer()
