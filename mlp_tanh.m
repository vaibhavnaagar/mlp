function a = mlp_tanh(z, grad=false)
%Activation function
%   a = MLP_Tanh(z) computes the activation of z.

e_z = exp(z);
e_neg_z = exp(-z);
a = (e_z - e_neg_z)./(e_z + e_neg_z);

if grad
  a = 1 - a.^2;
endif

clear e_z;
clear e_neg_z;
end
