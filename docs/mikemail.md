
# Mike Mail

The project is cool, actually this is an active research topic.
For getting into it more, you may look at literature around "Neural Architecture Search", proponents of this field, especially using EC for it, are e.g. Risto Mikkullainen.
Be sure to start with something simple indeed, so also with a simple agent (I would assume a simple network, not DQN), you can extend later on.
And yes, the game shall also be quite simple, all of these could work, you may try to obtain some fast running code so that you have more capacity to actually try out parameters.
There are some methods that have been used for optimization algorithms hyperparameter search, notable SMAC2, but this may not be very well suited to neural networks (don't know if somebody ever tried it).
Look out for literature, I myself was involved in SPOT (or SPO, SPOT is the software package) that is model-based in EGO style (EGO is a model based optimization method for expensive problems).

Some more information on EGO is explained below:

## EGO

EGO is a model-based optimization method for expensive problems.
It is based on the idea of fitting a surrogate model to the objective function and using this surrogate model to find the next point to evaluate.
The surrogate model is fitted to the objective function using a set of points that have already been evaluated.
The surrogate model is then used to find the next point to evaluate by optimizing an acquisition function.
The acquisition function is a function that balances exploration and exploitation.
It is used to find the next point to evaluate by finding the point that maximizes the acquisition function.
The point that maximizes the acquisition function is then evaluated and added to the set of points that have already been evaluated.
This process is repeated until a stopping criterion is met.
