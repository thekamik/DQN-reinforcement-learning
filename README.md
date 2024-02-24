# DQN-reinforcement-learning
Solving gymnasium pole card problem with DQN learning with PyTorch

1. This project was developed to implement a Deep Q-Network (DQN) for reinforcement learning tasks using PyTorch. <br />
This code serves as an example of agent training to interact with an environment, learn optimal actions, and maximize episode duration. <br />
In this project, the DQN algorithm is employed to solve the CartPole Gymnasium environment problem. <br />
[Gymnasium Cart Pole](https://gymnasium.farama.org/environments/classic_control/cart_pole/)

2. To run this code you need to install: <br />
   a.) Install the PyTorch library (this code was developed using pytorch 2.2.1 with CUDA 12.1 support on windows machine):<br />
   ```sh
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
   b.) Install gymnasium
   ```
   pip install gymnasium[classic-control]
   ```
   c.) Install matplotlib
   ```
   pip install matplotlib
   ```
   <br />

3. Here is chart with model performance: <br />
   ![Performance](https://github.com/thekamik/DQN-reinforcement-learning/blob/main/training_result.png)

   
4. If you find a bug or have an idea to improve the program, please e-mail me.
