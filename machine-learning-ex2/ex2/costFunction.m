function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
n = size(theta);
% You need to return the following variables correctly 

grad = zeros(size(theta));

% ====================== YOUR CODE HERE ===================================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
% Note: grad should have the same dimensions as theta
H(:,1) = sigmoid(theta'*X')';  % dim(m,1)
H(:,2) = ones(m,1) - H(:,1); % dim()
H = log(H);
y1 = ones(m,1)-y;
Y = [y y1]; 
C = Y.*H;
C = C(:,1) + C(:,2);
J = (-1/m)*sum(C);
%............................................................................

h = sigmoid(theta'*X')';
for j= 1:n
    grad(j) = sum((h-y)'*X(:,j));
end
grad = (1/m)*grad;

% =============================================================

end
