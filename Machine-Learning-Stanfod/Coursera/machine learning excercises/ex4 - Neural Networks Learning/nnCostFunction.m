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
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
matrix_y=[];     %transforming y into 0-1 y

for i = 1:num_labels
    matrix_y=[matrix_y,(y==i)];
end

%           Feedforward
%       Cost Function
a1=[ones(m,1),X];
z2=a1*Theta1';
a2=[ones(m,1),sigmoid(z2)];
z3=a2*Theta2';
a3=sigmoid(z3);
myOnes=ones(size(matrix_y));
Theta1_r=Theta1(:,2:size(Theta1,2));
Theta2_r=Theta2(:,2:size(Theta2,2));
squareTheta1=Theta1_r.*Theta1_r;
squareTheta2=Theta2_r.*Theta2_r;
regulirizedCost=lambda/(2*m)*((sum(sum(squareTheta1))+sum(sum(squareTheta2))));
J=1/m*sum(sum(-matrix_y.*log(a3)-(myOnes-matrix_y).*log(myOnes-a3)))+regulirizedCost;






%           Backpropagation

d3=a3-matrix_y;
Theta1(:,1)=zeros(size(Theta1,1),1);
Theta2(:,1)=zeros(size(Theta2,1),1);
Theta2_grad = 1/m * (a2'*(d3))'+lambda/m*Theta2;
z2=[ones(size(z2,1),1),z2];
d2=d3*Theta2.*sigmoidGradient(z2);
%d2=d3*Theta2.*a2.*(1-a2); also works!!
Theta1_grad = 1/m * (a1'*(d2))';
Theta1_grad=Theta1_grad(2:size(Theta1_grad),:) + lambda/m*Theta1;


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
