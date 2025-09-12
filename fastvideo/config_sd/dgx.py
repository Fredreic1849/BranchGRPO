import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def compressibility():
    config = base.get_config()

    config.pretrained.model = "CompVis/stable-diffusion-v1-4"

    config.num_epochs = 300
    config.save_freq = 50
    config.num_checkpoint_limit = 100000000

    # the DGX machine I used had 8 GPUs, so this corresponds to 8 * 8 * 4 = 256 samples per epoch.
    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    config.train.batch_size = 1
    config.train.gradient_accumulation_steps = 4

    # prompting
    config.prompt_fn = "imagenet_animals"
    config.prompt_fn_kwargs = {}

    # rewards
    config.reward_fn = "jpeg_compressibility"

    config.per_prompt_stat_tracking = {
        "buffer_size": 16,
        "min_count": 16,
    }

    return config

def hpsv2():
    config = compressibility()
    config.num_epochs = 300
    config.reward_fn = "hpsv2"

    # this reward is a bit harder to optimize, so I used 2 gradient updates per epoch.
    config.train.gradient_accumulation_steps = 8

    # the DGX machine I used had 8 GPUs, so this corresponds to 8 * 16 * 4 = 512 samples per epoch.
    config.sample.batch_size = 4

    # this corresponds to (16 * 4) / (4 * 8) = 2 gradient updates per epoch.
    config.train.batch_size = 4

    config.chosen_number = 16
    config.num_generations = 16
    return config

def hpsv3():
    config = compressibility()
    config.num_epochs = 300
    config.reward_fn = "hpsv3"

    # this reward is a bit harder to optimize, so I used 2 gradient updates per epoch.
    config.train.gradient_accumulation_steps = 8

    # the DGX machine I used had 8 GPUs, so this corresponds to 8 * 16 * 4 = 512 samples per epoch.
    config.sample.batch_size = 4

    # this corresponds to (16 * 4) / (4 * 8) = 2 gradient updates per epoch.
    config.train.batch_size = 4

    config.chosen_number = 16
    config.num_generations = 16
    return config


def get_config(name):
    return globals()[name]()
