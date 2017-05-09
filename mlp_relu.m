function a = mlp_relu(z, grad=false)
%Activation function
%   a = MLP_ReLU(z) computes the activation of z.
if grad
  a = zeros(size(z));
  a(find(z > 0)) = 1;
else
  a = max(0, z);
endif
end
