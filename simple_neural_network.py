from numpy import dot, random, array
class neural_network:
    def __init__(self):
        random.seed(1)
        self.weights = 2 * random.random((2, 1)) 

    def train(self, inputs, outputs, num):
        for _ in range(num):
            output = self.think(inputs)
            error = outputs - output
            adjustment = 0.01 * dot(inputs.T, error)
            self.weights += adjustment

    def think(self, inputs):
        return (dot(inputs, self.weights))

neural_network = neural_network()
inputs = array([[2, 3], [1, 1], [5, 2], [12, 3]])
outputs = array([[10, 4, 14, 30]]).T
neural_network.train(inputs, outputs, 200000)

print(neural_network.think(array([10, 10])))