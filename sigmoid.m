function g = sigmoid(z, grad=false)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

g = 1.0 ./ (1.0 + exp(-z));

if grad
  g = g.*(1-g);
endif
end
