import math
import random

# random.seed(0)

def rand(a, b):
    return (b - a) * random.random() + a


def make_matrix(m, n, fill=0.0):
    mat = []
    for _ in range(m):
        mat.append([fill] * n)
    return mat


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def sigmod_derivate(x):
    return x * (1 - x)


class DNN:
    def __init__(self):
        self.deep_len = 0
        self.layer = []
        self.cells = []
        self.weights = []
        self.correction = []

        
    def setup(self, layer, correct, learn):
        
        self.deep_len = len(layer)
        self.layer = layer
        self.layer[0] = self.layer[0] + 1
        
        # init cells
        for i in range(self.deep_len):
            cell = [1.0] * self.layer[i]
            self.cells.append(cell)
        
            
        # init weights
        for i in range(self.deep_len - 1):
            self.weights.append(make_matrix(self.layer[i], self.layer[i+1]))

        # random activate
        for i in range(self.deep_len - 1):
            for r in range(self.layer[i]):
                for c in range(self.layer[i + 1]):
                    self.weights[i][r][c] = rand(-1, 1)
        
        # init correction matrix
        for i in range(self.deep_len-1):
            self.correction.append(make_matrix(self.layer[i], self.layer[i+1]))
            
            
    def active_function(self, inputs, outputs, weight):
         
        for j in range(len(outputs)):
            total = 0.0
            for i in range(len(inputs)):
                total += inputs[i] * weight[i][j]
            outputs[j] = total
            
    def feed_forward(self, inputs):
        # activat input layer
# 
#     def feed_forward(self, inputs):
#         # activate input layer
#         for i in range(self.input_n - 1):
#             self.input_cells[i] = inputs[i]
#         # activate hidden layer
#         self.hidden_cells[0] = self.active_function(self.input_cells, self.input_weights)
#         hidden_out = []
#         for i in range(len(self.hidden_deep)-1):
#             hidden_out = self.active_function(self.hidden_cells[i], self.hidden_weights[i])
#         # activate output layer
#         self.output_cells = self.active_function(hidden_out, self.output_weights)
#         return self.output_cells
#     
#     def get_error(self, this_layer_cells, weights, up_deltas):
#         deltas = []
#         for i in range(len(this_layer_cells)):
#             error = 0.0
#             for o in range(len(up_deltas)):
#                 error += up_deltas[o] * weights[i][o]
#             deltas[i] = sigmod_derivate(this_layer_cells) * error
#         return deltas
#     
#     def update_weights(self, correction, learn, correct, weights, up_deltas, this_layer_cells):
#         new_weights = weights
#         new_correction = correction
#         for i in range(len(this_layer_cells)):
#             for o in range(len(up_deltas)):
#                 change = up_deltas[o] * this_layer_cells[i]
#                 new_weights[i][o] += learn * change + correction[i][o]
#                 new_correction[i][o] = change
#         return new_weights, new_correction
#     
#     def back_propagate(self, case, label, learn, correct):
#         # feed forward
#         self.feed_forward(case)
#         # get output layer error
#         output_deltas = [0.0] * self.output_n
#         for o in range(self.output_n):
#             error = label[o] - self.output_cells[o]
#             output_deltas[o] = sigmod_derivate(self.output_cells[o]) * error
#         
#         #get hidden last layer error
#         hidden_deltas = []
#         for i in range(len(self.hidden_deep)):
#             hidden_deltas.append([1.0] * self.hidden_deep[i])
#             
#         hidden_deltas[len(self.hidden_deep)-1] = self.get_error(self.output_cells, self.output_weights, output_deltas)
#         # update output weights
#         self.output_weights, self.output_correction = self.update_weights(self.output_correction, learn, correct, self.output_weights, output_deltas, self.hidden_cells)
# 
#         for i in range(len(self.hidden_deep)-1):
#             hidden_deltas[len(self.hidden_deep)- 2 - i] = self.get_error(self.hidden_cells[len(self.hidden_cells)-1-i], self.hidden_weights[len(self.hidden_weights)-1-i], hidden_deltas[len(self.hidden_deep) - 1 - i]) 
#         # update input weights
#         for i in range(self.input_n):
#             for h in range(self.hidden_n):
#                 change = hidden_deltas[h] * self.input_cells[i]
#                 self.input_weights[i][h] += learn * change + correct * self.input_correction[i][h]
#                 self.input_correction[i][h] = change
#         # get global error
#         error = 0.0
#         for o in range(len(label)):
#             error += 0.5 * (label[o] - self.output_cells[o]) ** 2
#         return error

if __name__ == '__main__':
    nn = DNN()
    nn.setup([2, 2, 1], 0, 0)

