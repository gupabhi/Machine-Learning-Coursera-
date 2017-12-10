function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = size(theta); % number of features
% You need to return the following variables correctly 
%J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
H(:,1) = sigmoid(theta'*X')';  % dim(m,1)
H(:,2) = ones(m,1) - H(:,1); % dim()
H = log(H);
y1 = ones(m,1)-y;
Y = [y y1]; 
C = Y.*H;
C = C(:,1) + C(:,2);
J = (-1/m)*sum(C);

J = J + (lambda/(2*m))*(sum(theta'*theta)-(theta(1)*theta(1))); % excluding contribution of theta0
%...........................................................................

h = sigmoid(theta'*X')';
grad(1) = (1/m)*sum((h-y)'*X(:,1));

for j= 2:n
    grad(j) = (1/m)*sum((h-y)'*X(:,j)) + (lambda/m)*theta(j);
end

% =============================================================
end
