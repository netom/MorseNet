from theano import tensor
from blocks.bricks import Linear, Rectifier, Softmax, MLP
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.roles import WEIGHT
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.initialization import IsotropicGaussian, Constant
from fuel.datasets import MNIST
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Flatten
from blocks.algorithms import GradientDescent, Scale
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing

# Feature and targets

x = tensor.matrix('features')
y = tensor.lmatrix('targets')

# Layers

mlp = MLP(activations=[Rectifier(), Softmax()], dims=[784, 100, 10], weights_init=IsotropicGaussian(0.01), biases_init=Constant(0))
mlp.initialize()

y_hat = mlp.apply(x)

# Cost

cg = ComputationGraph(y_hat)

W1, W2 = VariableFilter(roles=[WEIGHT])(cg.variables)

cost = CategoricalCrossEntropy().apply(y.flatten(), y_hat) + 0.005 * (W1 ** 2).sum() + 0.005 * (W2 ** 2).sum()
cost.name = 'cost_with_regularization'

# Dataset

mnist = MNIST(("train",))

data_stream = Flatten(DataStream.default_stream(
    mnist,
    iteration_scheme=SequentialScheme(mnist.num_examples, batch_size=1024)))

algorithm = GradientDescent(
    cost=cost, parameters=cg.parameters,
    step_rule=Scale(learning_rate=0.1)
)

mnist_test = MNIST(("test",))

data_stream_test = Flatten(
    DataStream.default_stream(
        mnist_test,
        iteration_scheme=SequentialScheme(mnist_test.num_examples, batch_size=4096)
    )
)

monitor = DataStreamMonitoring(
    variables=[cost], data_stream=data_stream_test, prefix="test"
)

main_loop = MainLoop(
    data_stream=data_stream, algorithm=algorithm,
    extensions=[monitor, FinishAfter(after_n_epochs=100), Printing()]
)

main_loop.run() 
