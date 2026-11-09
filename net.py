#!/usr/bin/env python3
import math
import random


class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data # tells python that a value object returns the data associated with it
        self.prev = set(_children)
        self.grad = 0.0 # default for gradient, no gradient.
        self._backward = lambda: None
        self.op = _op

    def __repr__(self):
        return f"Value(data={self.data})" #tells python to return a string of the value when a value object is called.

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other) # so we can add non-value items
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other) # so we can add non-value items
        out = Value(self.data * other.data, (self, other), "*")

        def _backward(): # basically, these functions update the value of the grads on the fly. So when the output grad is calcuated, it updates the grads of the previous nodes.
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only int/float powers supported"
        n = self.data
        p = other
        t = n**p
        out = Value(t, (self, ), "**")

        def _backward():
            self.grad += p*(n**(p-1)) * out.grad
        out._backward = _backward

        return out

    def __truediv__(self, other):
        return self * other**-1

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def tanh(self):
        n = self.data
        t = (math.exp(2*n) - 1)/ (math.exp(2*n) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        n = self.data
        t = math.exp(n)
        out = Value(t, (self, ), 'exp')

        def _backward():
            self.grad += t * out.grad
        out._backward = _backward

        return out

    def backward(self):
        topo, visited = [], set() # topo is a list of connections in reverse order of what needs have grad calculated first
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev: # does not run when there is no prev, for input nodes.
                    build(child) #the input nodes are added first
                topo.append(v) #adds the current node to the topology
        build(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


# BACKPROPOGATION IS SIMPLE : JUST CALCUATE HOW L CHANGES FOR EACH WEIGHT (GRADIENT) AND JUDGE THE WEIGHTS SO AS TO MINIMISE THE LOSS A LITTLE. NOT BIASES. NEURONS SUM
# INPUTS INTO IT, ADD A BIAS, AND SQUEEZE THE RESULT THROUGH AN ACTIVATION FUNCTION (KEEPING BETWEEN ZERO AND 1)

# USE CHAIN RULE LIKE SO: FOR ANY ELEMENTS X BEING COMBINED TO CREATE A NEW ELEMENT F. DL/DF * DF/DX = DL/DX. I.E. OUT.GRAD TIMES BY DF/DX. FOR TANH(X), DF/DX = 1 - TANH(X)^2

#i1, i2 = 0, 0
#w1, w2, w3, w4 = 4, 5, 1, 2
#w5, w6 = 3, 5
#n1 = i1 * w1 + i2 * w2
#n2 = i1 * w3 + i2 * w4
#l = n1 * w5 + n2 * w6
#dl/dw5 = n1
#dl/dw6 = n2
#etc

# TWO LAYER PERCEPTRON
class Neuron:
    def __init__ (self, nin): # nin = number of connections
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)] # random weight for each connection
        self.b = Value(random.uniform(-1, 1)) # random bias for the neuron

    def __call__(self, x): # x = list of activations going into neuron
        # zip(self.w, x) pairs each elements of the lists together
        activation = sum((wi*xi for xi, wi in zip(self.w, x)), self. b) # matrix multiplication essentially.
        out = activation.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]



class Layer: # produces nout neurons, and __call__ feeds the same input into all of them.
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons] # each neuron is called n, n(x) feeds the input into each neuron in the layer
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


L = Layer(2, 1) # 1 neuron with 2 inputs



class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))] # produces a list of layers of the sizes we want

    def __call__(self, x): #FORWARD PASS
        for layer in self.layers:
            x = layer(x) # FEEDS IN OUTPUTS OF PREVIOUS LAYER TO NEXT LAYER
        return(x)

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

L = MLP(2, [4, 4, 4, 1]) # 2 INPUTS, 2 layers of 3 neurons, output of 1 neuron

print(L.parameters())

xs = [
    [2,3,-1],
    [3,-1,0.5],
    [0.5,1,1],
    [1,1,-1]
]

ys = [1,-1,-1,1]




for y in range(0, 1000):
    for p in L.parameters():
        p.data -= p.grad * 0.01
        p.grad = 0

# FORWARD PASS
    ypred = [L(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred)) # our loss function, RMS

    loss.backward()
    print(loss)


# RESET GRAD TO ZERO AFTER BACKWARD PASS!!! OTHERWISE GRADS ACCUMMILATE AND RESULT ENDS UP EXPLODING!
