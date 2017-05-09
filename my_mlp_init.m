function [nhidden_layers, nnodes, actfun] = my_mlp_init()
% where
% nhidden_layers int 1x1
% nnodes int 1x nhidden_layers
% actfun string or int to index to activation function types

nhidden_layers = input("Enter number of hidden layers: ");
nnodes = [];
actfun = [];

for i=1:nhidden_layers
  fprintf('Layer-%d\n', i);
  nnodes = [nnodes input("Number of nodes: ")];
  actfun = [actfun menu("Select an activation function for this layer: ", "ReLU", "Tanh", "Sigmoid")];
end

actfun = [actfun menu("Select an activation function for Final(Output) layer: ", "ReLU", "Tanh", "Sigmoid")];

endfunction
