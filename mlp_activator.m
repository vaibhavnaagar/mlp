function a = mlp_activator(z, act_func=1, calc_grad=false)
%  Call Activation function depending on the value of act_func
%   a = MLP_ReLU(z) computes the activation of z.

switch act_func
  case 1                                      % Call ReLU
    a = mlp_relu(z, calc_grad);
  case 2                                      % Call Tanh
    a = mlp_tanh(z, calc_grad);
  otherwise
    a = sigmoid(z, calc_grad);                % Call sigmoid
endswitch
end
