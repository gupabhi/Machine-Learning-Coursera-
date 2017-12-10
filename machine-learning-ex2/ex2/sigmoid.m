function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 

% g = zeros(size(z));
% [a, b] = size(z);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
% format long g;

g = 1.0 ./(ones(size(z)) + exp(-1*z));
% === ==========================================================

end
