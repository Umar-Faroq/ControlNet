from share import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" # Use only GPU #3
#Umarfarooq
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint

# Configs
resume_path = './models/old_camus_control_sd15_ini.ckpt' # Path to the init model
batch_size = 4
logger_freq = 5000
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()

# Making a checkpoint callback that saves every model after a certain number of steps
call_checkpoint = ModelCheckpoint( # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html
    every_n_train_steps = 5000,  # Every 5000 steps
    save_top_k=-1  # All checkpoints are saved
)

dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger, call_checkpoint])


# Train!
trainer.fit(model, dataloader)
