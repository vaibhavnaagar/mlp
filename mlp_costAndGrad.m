function [Cost grad] = mlp_costAndGrad(weights, ...
                                   total_layers, ...
                                   nodes_per_layer, ...
                                   X, y, lambda, actfun)
% mlp_CostAndGrad Implements the neural network cost function for a N layer
% MLP which performs classification
%   It computes the cost and gradient of the MLP.
%   The returned parameter grad would be a "unrolled" vector of the
%   partial derivatives of the neural network.

% some useful variables
m = size(X, 1);
Cost = 0;
grad = [];
%================================================================================
%================================================================================
%                               Feedforward
%================================================================================
%================================================================================


previous_activation = X;                          % Input layer activation values
Activation_values = previous_activation';         % activation values: nodes x batch_size
regularized_term = 0;
previous_index = 0;
save_activations = [];
for L=2:total_layers
  current_index = previous_index + nodes_per_layer(L) * (nodes_per_layer(L-1) + 1);
  current_weights = reshape(weights((1 + previous_index):current_index), ...
                   nodes_per_layer(L), (nodes_per_layer(L-1) + 1));             % n_L x (n_(L-1)+1)
  current_activation = mlp_activator([ones(m, 1) previous_activation]*current_weights', actfun(L-1));
  Activation_values = [current_activation' ; Activation_values];              % save it for later in reverse order
  previous_activation = current_activation;
  previous_index = current_index;

  % For regularization
  regularized_term += sum(sum(current_weights(:, 2:end).^2));
end

% recode y of mx1 to Y mxK i.e y_i = [ 0 0 0 ...(ith)1....0]' where K is nmber of classes
Y = zeros(m, nodes_per_layer(end));
Y(sub2ind(size(Y), [1:m], y')) = 1;
% Another method for above line:
%Y((1:m)' + (y-1)*m) = 1;    % (1:m)' + (y-1)*m  converts all y(mx1) values to linear index for Y (mxK)

Cost = mlp_costFunction(Y, previous_activation, m, "squared_error");
Cost += (lambda/(2*m))*regularized_term;

%================================================================================
%================================================================================
%                           Backpropagation
%================================================================================
%================================================================================

previous_del = (previous_activation - Y);%.*mlp_activator(previous_activation, actfun(end), true);                                         % (m x 10)
%previous_index = nodes_per_layer[end] * (nodes_per_layer[end-1]+1);
row_index = size(previous_activation, 2);
previous_index = size(weights, 1) + 1;

for L=fliplr(1:(total_layers-1))
  current_index = previous_index - (nodes_per_layer(L) + 1) * nodes_per_layer(L+1);
  current_weights = reshape(weights(current_index:(previous_index-1)), ...
                        nodes_per_layer(L+1), (nodes_per_layer(L) + 1));
  current_activation = Activation_values((row_index+1):(row_index+nodes_per_layer(L)), :)'; % m x nodes_L

  new_del = ((previous_del*current_weights)(:,2:end)).*mlp_activator(current_activation, actfun(L), true);  % m x nodes_L
  new_grad = (previous_del'*[ones(m, 1) current_activation])/m + current_weights*lambda/m;
  new_grad(:, 1) = new_grad(:, 1) + current_weights(:, 1)*lambda/m;
  grad = [new_grad(:) ; grad];

  previous_index = current_index;
  row_index += nodes_per_layer(L);
  previous_del = new_del;

end
end
