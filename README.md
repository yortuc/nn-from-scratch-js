# Neural network from scratch with js

### Why?
Implementing a very simple but fully functional nn is not that at first time easy imho. 
To grasp every aspect of the computing behind a nn, i developed every single line of code, including mathematical functions like matrix multiplication, softmax and so on.   

### Testing

1. All functions are `pure` in this project. So, testing these functions is relatively easy. Almost every function has a test function like `test_softmax()`.

2. Test the neural network with a single data point and an arbitrary target value.

```
	test input:
		x = [[1, 2, 3]];

	test target:
		y = [[0.25, 0.75]];
```

Since a neural network is an universal approximator, it should compute some weights to produce this output respect to input.

Run `test_train_nn()` and check convergence.


### Iris Classification Problem

	features	
	  x           h           y      output
	-----       -----      ------   --------
	x1          n1.1 
	x2          n1.2        n2.1 
	x3          n1.3        n2.2        S
	x4          n1.4        n2.3
    -----       -----      -------  ---------

- first layer has 4 neurons (iris dataset has 4 features for input points: (sepal length, sepal width, petal length, and petal width)
- hidden layer's neuron count can be changed
- output layer has 3 neurons for output (one-hot representation for setosa, versicolor, and virginica)

### Computing sums

	Layer1: Z1

	        data points			  hidden layer neuron weights
	--------------------------    ----------------------

	point1 | x11 x12 x13 x14 |   | w11  w21  w31  w41 |   | z11 z12 z13 z14 |
 	point2 | x21 x22 x23 x24 |   | w12  w22  w32  w42 |   | z21 z22 z23 z24 |
	point3 | x31 x32 x33 x34 | ⊗ | w13  w23  w33  w43 | = | z31 z32 z33 z34 |
	point4 | x41 x42 x43 x44 |   | w14  w24  w34  w44 |   | z41 z42 z43 z44 |
	  .                                                             .
	  .                                                             .
	  .                                                             .

	z11 = w11*x1 + w12*x2 + ... + b1
	a1 = activation_function(xW + b)

	Layer2: Z2

			z1			 output layer neuron weights
	------------------     ----------------------

	| z11 z12 z13 z14 |   | w11 w21 w31 |   | y11 y12 y13 |   | 1 0 0 |    setosa
	| z21 z22 z23 z24 |   | w12 w22 w32 |   | y21 y22 y23 |   | 1 0 0 |    setosa
 	| z31 z32 z33 z34 | ⊗ | w13 w23 w33 | = | y31 y32 y33 | > | 0 0 1 | >  virginica
	| z41 z42 z43 z44 |   | w14 w24 w34 |   | y41 y42 y43 |   | 0 1 0 |    versicolor
	         .                                      .             .            .
	         .                                      .             .            .
	         .                                      .             .            .

	Layer2 a2 is actually the output of the neural network.

### Activation functions 
`tanh` for hidden layer and `softmax` for output layer is used.

### Backpropagation

As James McCaffrey stated beautifuly, backpropagation flows like this: 

```
	loop until some exit condition
	 compute output values
	 compute gradients of output nodes
	 compute gradients of hidden layer nodes
	 update all weights and bias values
	end loop
```

We have 2 gradients for output and hidden layers.

**Output layer**: The gradient of output node is the difference between the computed output value and the
desired value, multiplied by the calculus derivative of the activation function used by the output
layer.

```js
	const grad_output = comp_prod(diff_output, deriv_a2)
```

**Hidden layer**: Computing the values of the hidden node gradients uses the values of the output node
gradients. The gradient of a hidden node is the derivative of its activation function times the sum
of the products of "downstream" output gradients and associated
hidden-to-output weights. 

```js
	const sum = had_product(grad_output, mat_transpose(w2));
	const deriv_a1 = hyperbolic_tangent_grad(a1);
	const grad_hidden = comp_prod(sum, deriv_a1);
```

### Training

Common training methods for nns:

1. Batch training
2. Incremental training
3. Minibatch training

Incremental training and batch training implemented. In this very case, and also like experts stated, 
incremental training seems more robust and converges faster.

**Shuffling** :
Using incremental training, it's important to make sure that the neural netwok accesses data randomly in every loop.
In my experiments, i also experienced the shuffle effect.

### Result

```
	------- accuracy --------
	72 correct  3 wrong
	%  96
	------ validation -------
	error:  0.10099364087457778
	34 correct  3 wrong
	%  91.89189189189189
```
