Write a system prompt for an AI agent.

The agent will be shown recorded data from a single unit (one neuron) of a trained neural network: either windows of text with the unit's activation marked on tokens, or a grid of images with the unit's activation on each. It must say what the unit does. The user message will specify a fixed JSON reply: a class (one thing / one main thing plus related extensions / two or more unrelated things / the data is not enough to tell), a short explanation, the evidence, what data is missing, and a confidence. In a second turn it is shown 20 new inputs and must predict, for each, whether the unit fires strongly or barely at all.

The agent has no tools and cannot run experiments: it reads the data and answers. Its answers will be checked against exact ground truth about what the unit does, including its predictions on the new inputs.
