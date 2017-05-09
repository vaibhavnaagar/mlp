function [cost] = mlp_costFunction(Y, output, num_examples, ...
                                   func="squared_error")

cost = 0;
assert(size(Y) == size(output));

if strcmp(func, "squared_error")
  cost = sum(sum((Y - output).^2))/(2*num_examples);
elseif strcmp(func, "cross_entropy")
  cost = sum(sum(-Y.*log(output) - (1-Y).*log(1 - output)))/num_examples;
else
  body
end


end
