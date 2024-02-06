# multimodal_contrastive

```
#set up environment:
conda env create -f env.yml

#install develop version of multimodal_contrastive module
pip install -e .

```


# for Pretrain

1. set up config
modify multimodal_contrastive/configs/pretrain.yaml file to change model and data set as needed

2. run command
```
# for training
bsub -J test -o output.%J -e output.%J -gpu "num=1" -n 16 -q long -sla gRED_resbioai_gpu
"python multimodal_contrastive/multimodal_contrastive/pretrain.py"
```

### set up config
add configuration files or modify/use existing configuration file under configs/ to train as needed

some configuration files:
1. configs/puma_smg_gmc.yaml: 3modality GMC model use PUMA dataset
2. configs/puma_sm_gmc.yaml: 2modality GMC model use PUMA dataset
3. configs/puma_sm_clip.yaml: 2modality CLIP model use PUMA dataset
4. configs/jump_sm_gmc.yaml: 2modality GMC model use JUMP dataset
5. configs/jump_sm_clip.yaml: 2modality CLIP model use JUMP dataset

### run command
```
# for training
bsub -J test -o output.%J -e output.%J -gpu "num=1" -M 4G -n 16 -q long -sla gRED_resbioai_gpu "HYDRA_FULL_ERROR=1 CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 python pretrain.py --config-name jump_sm_clip.yaml"
```

# for Sweeps

1. generate sweep path:
```
wandb sweep configs/sweeps/sweeps.yaml
```

3. copy sweep path to multimodal_contrastive/sweep.bsub
```
#!/bin/bash -i
#BSUB -J sweeps[1-10]
#BSUB -q long
#BSUB -n 18
#BSUB -gpu "num=1"
#BSUB -o /home/luz21/scratch/logs/output.%I
#BSUB -e /home/luz21/scratch/logs/output.%I

sweep_path=<sweep_path>

wandb agent ${sweep_path}
```

3. submit bsub script
```
bsub < sweep.bsub
```


