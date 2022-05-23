# Fast PyTorch Reinforcement and Imitation Learning (Work in progress)
I've implemented and tested couple of RL and IL algorithms in my previous
[*project*](https://github.com/CherryPieSexy/imitation_learning),
but it is build on top of the gym async vector env,
which is not quite fast solution for gathering experience from multiple environments.

In this project I am trying to make a similar framework (maybe not as general as the previous one)
but focusing on the speed of data collecting and training.

Design is not one of my strength, but I tried to make a descriptive visualization
of the current structure of the parallelized experience collection from independent environments:
![scheme](imgs/parallelism_scheme.png)

I am inspired by this awesome project by Alex Petrenko: https://github.com/alex-petrenko/sample-factory

Any suggestions or contributions are welcome.
Feel free to contact me directly through the email or the telegram.

TODO:
- [x] Create readme
- [ ] Code environment and model workers, throughput process
- [ ] Test speed on the CartPole env, add results to the table
- [ ] Make automatic detection of optimal throughput parameters
(number of envs n, number of env workers N, number of model workers M)
- [ ] Create some data buffer to store rollouts in
- [ ] Add basic trainer A2C to test if this approach works
- [ ] Add advanced trainer (PPO)
