function p = mlp_predict(weights, X, total_layers, nodes_per_layer, actfun, save_=false)
% Predict the label of an input given a trained MLP

m = size(X, 1);
p = zeros(size(X, 1), 1);

previous_activation = X;                          % Input layer activation values
previous_index = 0;

for L=2:total_layers
  current_index = previous_index + nodes_per_layer(L) * (nodes_per_layer(L-1) + 1);
  current_weights = reshape(weights((1 + previous_index):current_index), ...
                   nodes_per_layer(L), (nodes_per_layer(L-1) + 1));             % n_L x (n_(L-1)+1)
  current_activation = mlp_activator([ones(m, 1) previous_activation]*current_weights', actfun(L-1));
  previous_activation = current_activation;
  previous_index = current_index;

  if (L == total_layers-1) && (save_ == true)
    size(current_activation)
    save featuremap_b_16.mat current_activation;
  endif
end
assert(current_activation == previous_activation)
[dummy, p] = max(current_activation, [], 2);
p(find(p == 10)) = 0;                           % map 10 -> 0
endfunction
