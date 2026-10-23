This project is based on an article on Denny's Blog 
https://dennybritz.com/posts/wildml/implementing-a-neural-network-from-scratch/ 

Below are some notes i copied from the blog:

> During training a neural network, The number of nodes in the input layer is determined by the dimensionality of our data, 2. Similarly, the number of nodes in the output layer is determined by the number.
> softmax is a way to convert raw scores to probabilities
> Variations such as SGD (stochastic gradient descent) or minibatch gradient descent typically perform better in practice. So if you are serious you’ll want to use one of these, and ideally you would also decay the learning rate over time.

>what is regularization strength?
Regularization strength controls how strongly training penalizes complex models, usually by discouraging large weights. This helps reduce overfitting—memorizing training data instead of learning patterns that work on new data.
- Too high: the model may be too constrained and underfit.
- Too low: it may overfit.
- With L2 regularization: total_loss = prediction_loss + λ * sum(weights²), where λ is the regularization strength.


>what is learning rate?
Learning rate controls how big a step a neural network takes when updating its weights to reduce error.
- Too high: it may overshoot good weights or fail to converge.
- Too low: learning is slow and may make little progress.
- In an update like weight -= learning_rate * gradient, it scales the gradient.