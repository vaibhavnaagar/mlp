function checkNNGradients2(lambda)
%CHECKNNGRADIENTS Creates a small neural network to check the
%backpropagation gradients
%   CHECKNNGRADIENTS(lambda) Creates a small neural network to check the
%   backpropagation gradients, it will output the analytical gradients
%   produced by your backprop code and the numerical gradients (computed
%   using computeNumericalGradient). These two gradient computations should
%   result in very similar values.
%

input_layer_size = 3;
hidden_layer_size = 5;
num_labels = 3;
m = 5;

% generate some 'random' test data
Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size);
Theta2 = debugInitializeWeights(num_labels, hidden_layer_size);

% Reusing debugInitializeWeights to generate X
X  = debugInitializeWeights(m, input_layer_size - 1);
y  = 1 + mod(1:m, num_labels)';


% Unroll parameters
nn_params = [Theta1(:) ; Theta2(:)];

act = [2 2];
% Short hand for cost function
costFunc = @(p) mlp_costAndGrad(p, 3, [3 5 3], ...
                               X, y, lambda, act);

[cost, grad] = costFunc(nn_params);
numgrad = computeNumericalGradient(costFunc, nn_params);

% Visually examine the two gradient computations.
disp([numgrad grad]);
fprintf(['(Left- Numerical Gradient, Right-Analytical Gradient)\n\n']);

% Evaluate the norm of the difference between two solutions.
diff = norm(numgrad-grad)/norm(numgrad+grad);

fprintf(['\nRelative Difference (Numerical and analytical gradients): %g\n'], diff);

end
