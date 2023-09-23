import os
from pathlib import Path

class Saver:
    def __init__(self, config, template="{}_{}_{}.{}"):
        self.config = config
        self.template = template
        os.makedirs(self.config.train.ckpt_dir, exist_ok=True)
    
    def get_save_path(self, prefix="exp", suffix="pth", epoch=100):
        if suffix == "pth":
            ckpt_dir, ckpt_path = self._get_pth_save_path(epoch=epoch, prefix=prefix)
        elif suffix == "npy":
            ckpt_dir, ckpt_path = self._get_npy_save_path(epoch=epoch, prefix=prefix)
        
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        return ckpt_path

    def _get_pth_save_path(self, prefix="exp", epoch=100):
        ckpt_dir = os.environ.get("SLURM_TMPDIR", self.config.train.ckpt_dir)
        ckpt_path = os.path.join(
            ckpt_dir,
            self.template.format(
            prefix,
            self.config.valid.algorithm if self.config.train.mode == 'eval' else self.config.train.algorithm,
            epoch, "pth")
        )
        return ckpt_dir, ckpt_path

    def _get_npy_save_path(self, prefix="exp", epoch=100):
        main_dir = os.environ.get("SLURM_TMPDIR", self.config.train.save_dir) if "precache" in prefix else self.config.train.save_dir
        model_name = self.config.train.model.replace("proj", "").replace("feat", "")
        ckpt_dir = os.path.join(main_dir, model_name)

        if self.config.train.mode == "valid":
            algorithm = self.config.valid.algorithm
        else:
            algorithm = self.config.train.algorithm

        # add num_augmentations to ckpt_dir
        ckpt_dir = os.path.join(ckpt_dir, "{}_augs".format(self.config.train.num_augmentations))
        ckpt_dir = self._get_ckpt_dir_path_from_algorithm(ckpt_dir)
        
        # template : prefix__{precache/algorithm_epoch}__seed42.npy
        ckpt_path = os.path.join(
            ckpt_dir,
            "{}__{}__{}.{}".format(
                prefix,
                "" if "precache" in prefix else "{}_{}".format(algorithm, epoch),
                "seed{}".format(self.config.train.seed),
                "npy",
            ),
        )

        return ckpt_dir, ckpt_path

    def _get_ckpt_dir_path_from_algorithm(self, ckpt_dir):
        if self.config.train.mode == "valid":
            algorithm = self.config.valid.algorithm
        else:
            algorithm = self.config.train.algorithm
        if algorithm == "BTwins":
            ckpt_template =  "lambd_{:.6f}".format(
                    self.config.train.lambd,
                 )
        elif algorithm == "SimCLR":
            ckpt_template = "temp_{:.3f}".format(
                    self.config.train.temperature,
                    )
        ckpt_template = "{}_pdim{}_bsz{}_lr{}_wd{}".format(
                    ckpt_template,
                    self.config.train.projector_dim,
                    "_no_autocast" if not self.config.train.use_autocast else "",
                    self.config.train.lr,
                    self.config.train.weight_decay,
                )
        if self.config.train.mode == "valid":
            return os.path.join(
                ckpt_dir,
                ckpt_template,
                "augs{}_eval".format(self.config.valid.num_augmentations),
            )
        else:
            return os.path.join(
                ckpt_dir,
                ckpt_template,
                "augs{}".format(self.config.train.num_augmentations),
            )
        
        



                                                                




            

