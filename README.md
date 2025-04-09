## AIviator
AIviator is an AI for the popular board game
[Sky Team](https://boardgamegeek.com/boardgame/373106/sky-team) trained with deep reinforcement
learning.

### 🚀 Usage
Install the dependencies: `pip install -r requirements.txt`

To play with the AI or watch it play, download the model checkpoint from [here](https://drive.google.com/file/d/14RZzMjbelV9ex1-3GVBN-87UWBRagl56/view?usp=sharing) and extract the contents into the `./data/aiviator` directory. Then run `python play_game.py`.

To train an your own model use `learning.py`.

### ✈️ Playable maps and abilities
AIviator can play all of the red and black airports except Haneda (as the intern rule is not
implemented). It can play with the Anticipation, Control, Mastery, and Working Together abilities.

### 🤖 How good is the AI?
AIviator is very good at the game -- definitely better than my partner and me! With the right abilities, it can beat Paro over 90% of the time (although of course it doesn't need to worry about the time limit). Here are its win rates in various settings:

| Airport  | WT + Mastery | WT + Control | WT + Anticipation | Mastery + Control | Mastery + Anticipation | Anticipation + Control |
|----------|----------------------------|-----------------------------|----------------------------------|-------------------|------------------------|-------------------------|
| PBH      | 92.1%                      | 90.0%                       | 87.1%                            | 89.5%             | 87.9%                  | 84.8%                   |
| KEF      | 96.6%                      | 96.6%                       | 94.9%                            | 95.7%             | 93.7%                  | 94.0%                   |
| KUL      | 98.8%                      | 98.3%                       | 98.4%                            | 98.2%             | 98.2%                  | 97.4%                   |
| PBH_RED  | 99.1%                      | 98.8%                       | 98.9%                            | 99.2%             | 98.8%                  | 98.8%                   |
| GIG      | 98.6%                      | 98.3%                       | 98.0%                            | 97.7%             | 97.3%                  | 96.7%                   |
| TGU      | 97.5%                      | 96.7%                       | 96.3%                            | 96.2%             | 95.9%                  | 95.5%                   |
| OSL      | 94.6%                      | 96.1%                       | 91.6%                            | 94.1%             | 91.4%                  | 91.9%                   |


### 🧠 How does the AI play?
Here are some observations:
* Of the special abilities, it seems that working together > control ≈ mastery > anticipation.
* PBH is the hardest airport. Surprisingly, the second hardest is OSL, a red level.
* Across rounds, AIviator generally removes planes first before worrying about systems.
  Within a round, it generally does radio, then axis, then engines, then other systems. However,
  there is quite a bit of variance depending on the situation.
* It usually uses rerolls at the beginning of the round. The player deciding to use the reroll
  will often reroll all 4 dice.
* Similarly, it uses working together early in the round.
  Often the swap is used to control the plane's tilt (i.e. you pass a die you want your
  ally to put in axis). It uses working together on about half of the rounds.
* It doesn't use brakes much, often only getting them to 2 and relying on coffees to reduce its
  speed.

The AI cannot directly communicate with its partner at all. However, it could learn indirect communication strategies based on die placements ("If I place a die on axis instead of radio, it means I can't remove the plane in front of us")

### 🛠 Technical details
The model is trained using PPO, with general advantage estimation and entropy regularization. The model architecture is a transformer over a length-21 sequence. The first token contains general information about the game state. The other 20 tokens consist of all combinations of the 4 dice and 5 possible coffee values (between -2 and 2 -- it never uses 3 coffees) for each die. Each die token emits a logit for every slot it could be placed in as well as a logit on whether the dice should be rerolled. The first token emits an additional logit on whether a reroll should be used. These logits are concatenated to get a distribution over possible actions for the turn. I tried a few other architectures, but this one worked the best.

The biggest challenge wasn't the actual RL, but designing a system to collect trajectories efficiently. Before starting the project, I assumed that the compute cost of simulating the game would be tiny compared to running the model. In fact, when running on a GPU simulating the game quickly becomes the bottleneck. A classic solution is to asynchronously collect trajectories on many CPUs and then train on GPU. However, because Sky Team is a simple enough game, I opted for another approach: I implemented the game rules in pure Jax so the environment could be jitted/vmapped and then run on the GPU in batches. This worked great: during training, running the environment takes up <5% of the compute.

### ⚠️ Disclaimer
This is an unofficial AI project inspired by *Sky Team*, a board game by Scorpion Masqué.
All game mechanics and intellectual property rights belong to their respective owners.
This implementation is for educational and non-commercial purposes only.