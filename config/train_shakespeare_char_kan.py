# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-shakespeare-char-kan'
eval_interval = 50 # keep frequent because we'll overfit
eval_iters = 20
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt-kan'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 4
block_size = 32 # context of up to 256 previous characters

width: tuple = (16, 16, 16, 16)
grid: int = 15
k: int = 3
noise_scale: float = 0.1
noise_scale_base: float = 0.1
base_fun: torch.func = torch.nn.SiLU()
symbolic_enabled: bool = True
bias_trainable: bool = True
grid_eps: float = 1.0
grid_range: tuple = (-1, 1)
sp_trainable: bool = True
sb_trainable: bool = True
seed: int = 0
vocab_size: int = int(2 ** 16)

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
device = 'mps'  # run on cpu only
compile = False # do not torch compile the model
