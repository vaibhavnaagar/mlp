function W = randInitializeWeights(L_in, L_out)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections in order to break the symmetry.
%    W should be set to a matrix of size(L_out, 1 + L_in) as
%   the first column of W handles the "bias" terms

%--------------------------------------------------------------------
%  For range of weights, the normalization factor: sqrt(6)/(sqrt(L_in + L_out))
% is taken from the paper-
%     Y.Bengio, X. Glorot, Understanding the difficulty of training deep feedforward neuralnetworks, AISTATS 2010
%  at url: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
%------------------------------------------------------------------------

  epsilon_init = sqrt(6)/(sqrt(L_in + L_out));
  W = rand(L_out, 1+L_in)*2*epsilon_init - epsilon_init;

end
