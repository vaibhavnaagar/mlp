function checkMLPGradients(weights, ...
                          total_layers, ...
                          nodes_per_layer, ...
                          X, y, lambda, actfun)

%   CHECKNNGRADIENTS(lambda), it will output the analytical gradients
%   produced by your backprop code and the numerical gradients (computed
%   using computeNumericalGradient).
%

costFunc = @(p) mlp_costAndGrad(p, total_layers, nodes_per_layer, ...
                                X, y, lambda, actfun);


[cost, grad] = costFunc(weights);
numgrad = computeNumericalGradient(costFunc, weights);


disp([numgrad(1:100) grad(1:100)]);
fprintf(['(Left Numerical Gradient, Right-Analytical Gradient)[for first 100 weights]\n\n']);

% Evaluate the norm of the difference between two solutions.
diff = norm(numgrad-grad)/norm(numgrad+grad);

fprintf('Relative Difference (between numerical and analytical gradients): %g\n', diff);

end
