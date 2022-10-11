# Run example
```
python train.py --algo ppo --env ReachEnv-v0 --yaml-file hyperparameters/ppo.yml
```

```
python train.py --algo sac --env ReachEnv-v0 --yaml-file hyperparameters/sac.yml --seed 42 --vec-env subproc -P --eval-episodes 1 --track --wandb-project-name n-dim-reach 
```