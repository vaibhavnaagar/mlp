function ret_weights = mlp_train(weights, ...
                                   total_layers, ...
                                   nodes_per_layer, ...
                                   X, y, learning_rate, lambda, actfun, method="gd_momentum", X_valid, y_valid)

% parameters
mu = 0.9;
fudge_factor = 1e-6;                     % smoothing term to avoid division by zero

batch_size = 16
m =length(y);
num_iterations = 5;
%J_history = zeros(num_iterations, 1);
index_vector = randperm(m);
delta_w = zeros(size(weights));
accumulated_grad = zeros(size(weights));

cost_history = [];
valid_history = [];

ret_weights = weights;
steps = 0;
for iters=1:num_iterations
  for idx=1:batch_size:m
%      fprintf("Index: %d ==> ",idx);
      [Cost grad] = mlp_costAndGrad(ret_weights, total_layers, nodes_per_layer, ...
                          X(index_vector(idx:min((idx+batch_size-1), m)), :), y(index_vector(idx:min((idx+batch_size-1), m))), lambda, actfun);

      switch method
        case "gd_momentum"                                % Gradient descent with momentum
          delta_w = learning_rate*grad + mu*delta_w;
          ret_weights = ret_weights - delta_w;
        case "adaGrad"                                    % AdaGrad
          accumulated_grad += grad.^2;
          ret_weights = ret_weights - learning_rate*grad./(sqrt(accumulated_grad) + fudge_factor);
        otherwise                                         % Standard gradiest descent
          ret_weights = ret_weights - grad*learning_rate;
      endswitch

%      Cost
      cost_history = [cost_history; Cost];
      if mod(steps, 100) == 0
        [valid_cost valid_grad] = mlp_costAndGrad(ret_weights, total_layers, nodes_per_layer, ...
                            X_valid, y_valid, lambda, actfun);
        fprintf("validation cost: %f\n", valid_cost);
        valid_history = [valid_history; steps Cost valid_cost];
      endif
      steps += 1;

  end
  delta_w = zeros(size(weights));
  accumulated_grad = zeros(size(weights));
  index_vector = randperm(m);
  fprintf("Cost: epoch %d: %f\n",iters, Cost);
end

save cost_b_16.mat cost_history;
save valid_b_16.mat valid_history;
endfunction
