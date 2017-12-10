function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
%J = 0;
%Theta1_grad = zeros(size(Theta1));
%Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% implemeting forward propagation
X = [ones(size(X,1),1) X]; %dimension 5000X401

z2 = X*Theta1';  % row contains activation value of an input dim(5000X26) 
a2 = sigmoid(z2); % dim(5000X26) 
A2 = [ones(size(a2,1),1) a2]; % dim 5000X26
z3 = A2*Theta2'; %  dim 5000X10
h = sigmoid(z3); % dim 5000X10
%disp(h);
% write now I'm treating y as 5000X10
H1 = h; % dim 5000X10
H1 = log(H1);
H2 = 1 - h; % dim 5000X10
H2 = log(H2);
%H = [H1 H2]; % dim 5000X20
%H = log(H); % dim 5000X20

%..........................................................................
% map y (dim 5000X1) to y (dim 5000X10))
temp2 = zeros(m, num_labels); % (dim 5000X10)
for i = 1:m
   index = y(i,1); 
   temp2(i, index) = 1; 
end   
y = temp2;

%..........................................................................
y1 = y; % dim 5000X10
y2 = (1-y1); % dim 5000X10
%Y = [y1 y2]; % dim 5000X20

J = -(1/m)*sum(sum(y1.*H1)+sum(y2.*H2))  +  ...
    (lambda/(2*m))*(sum(sum(Theta1(:,2:size(Theta1,2)).^2))+ ...
        sum(sum(Theta2(:,2:size(Theta2,2)).^2))); % dim 1X1

% =========================================================================

% implementing backpropagation
% z2 dim 5000 25
Delta3 = zeros(num_labels, size(A2,2));  % dim 10 26
Delta2 = zeros(size(z2,2), size(X,2)); % dim 25 401
for i=1:m
    delta3 = (h(i,:)-y(i,:))'; % dim 10X1
    delta2 = (Theta2'*delta3).*([0 sigmoidGradient(z2(i,:))])'; % dim 26x1
    Delta3 = Delta3 + delta3*A2(i,:); 
    Delta2 = Delta2 + delta2(2:size(delta2))*X(i,:);
end

M1 = [zeros(size(Theta1,1),1) Theta1(:,2:size(Theta1,2))];
M2 = [zeros(size(Theta2,1),1) Theta2(:,2:size(Theta2,2))];
D2 = (1/m)*Delta2 + lambda*M1;
D3 = (1/m)*Delta3 + lambda*M2;

Theta1_grad = D2;
Theta2_grad = D3;
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end

