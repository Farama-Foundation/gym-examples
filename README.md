# Run example
```
python train.py --algo ppo --env ReachEnv-v0 --yaml-file hyperparameters/ppo.yml
```


```
python train.py --algo sac --env ReachEnv-v0 --yaml-file hyperparameters/sac.yml --env-kwargs n_dim:3 --seed 42 -P --eval-episodes 3 --eval-freq 20000
```
With WandB
```
python train.py --algo sac --env ReachEnv-v0 --yaml-file hyperparameters/sac.yml --env-kwargs n_dim:2 reward_shaping:False --seed 42 -P --eval-episodes 3 --eval-freq 20000 --track --wandb-project-name n-dim-reach 
```