# Surprise-based exploration with a model-based architecture
This is the code repository for my thesis.

#### Contents
- Experiments for combining an autoencoder with a prediction model in the latent space in Atari environments
- Experiments for extracting uncertainty in deep neural networks (MC dropout and bootstrap) in simple regression task
- A3C implementation with the surprise-based exploration method (running Atari games)

#### TODO
##### TODO Architecture
- Try using the compressed latent space from the CNN in the DQN/A3C
- Try to take the difference between frame 1 and frame 4 and feed it to the autoencoder
- Try to make a decoder that decodes the latent space from the CNN in the DQN/A3C. Prediction net can then use one of the first layers in the decoder.
- Try with different dropout rate for the prediction network (0.05 --> from the Deep PILCO paper)

##### TODO DQN
- Implement prioritized experience replay (dynamics model will benefit from this as well)
- Implement saving of images from the autoencoder (and possibly the prediction model) during a run
- Change autoencoder to take one frame at the time and then concatenate the latent vectors for the 4 consecutive frames


##### TODO A3C
- Implement my architecture for surprise-based exploration