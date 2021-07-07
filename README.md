# Examining the performance Temporal Difference learning procedure

## Temporal Difference (TD) learning
Temporal difference learning refers to a class of model-free reinforcement learning algorithms that learn by bootstrapping from the current estimate of the value function. The method was first introduced by Sutton in 1988. In his paper, Sutton stated that temporal-differences methods can produce better predictions with less memory and peak computation compared to supervised learning methods. The author illustrated his statements through two computational experiments, showing that temporal-differences learning procedure outperform supervised learning procedure in learning to predict in both cases. In our code and paper, we replicate the experiments to examine the performance of temporal-differences methods in prediction learning when using different data set and key parameters. Results indicates that temporal-difference learning procedures indeed learn faster than supervised learning, outperforming it in producing predictions. My paper reexamines Sutton’s (1988) study on temporal-differences methods’ learning performance by replicating his findings.


Sutton (1988): [Click Here](https://www.researchgate.net/publication/225264698_Learning_to_Predict_by_the_Method_of_Temporal_Differences)

Replication Analysis: [Click Here](https://github.com/anhducvu-bot/Temporal_Difference_Experiment/blob/main/Temporal%20Difference%20Replication%20-%20Anh%20Vu.pdf)

## A small note: The bootstraping nature of TD method and the nature of expectation/happiness in human: 
TD algorithms works by bootstrapping, i.e., instead of trying to calculate the total expected rewards, the method only calculates the immediate rewards and the reward expectation at the next step (hence "bootstrapping"). As an example, imagine two person: the first person is going to the bank to cash out his $100 check, and the second person who already cashed out the $100 check, but forgot that he already did and still is going to the bank. According to TD methods, when both got the 100$ bill, the second person will feel happier than the first person he bootstrapped his expectation that he hasn't cashed out the check, but in fact did, hence he received a higher reward value than the first person. Interestingly, this method has been shown to confirm the reward prediction error theory of dopamine of animals. I found it very fascinating that when we apply a simple method that is consistent with how animals release dopamine, the result is a very powerful model-free reinforcement learning algorithms that can play backgammon, land spaceship (DQN agent performing OpenAI's Lunar Lander), and even beat the human expert in the game of go (AlphaGo's Deepmind). It's intersting to think that the nature of intelligence is actually just maximizing rewards (dopamine in our case). 
