function numgrad = computeNumericalGradient(J, theta)
% COMPUTENUMERICALGRADIENT Computes the gradient using "finite differences"
%   and gives us a numerical estimate of the gradient.
%   numgrad = COMPUTENUMERICALGRADIENT(J, theta) computes the numerical
%   gradient of the function J around theta.


numgrad = zeros(size(theta));
perturb = zeros(size(theta));
ae = 1e-4;
for p = 1:numel(theta)
    % Set perturbation vector
    perturb(p) = ae;
    loss1 = J(theta - perturb);
    loss2 = J(theta + perturb);
    % Compute Numerical Gradient
    numgrad(p) = (loss2 - loss1) / (2*ae);
    perturb(p) = 0;
end

end
