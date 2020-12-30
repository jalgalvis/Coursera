function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    prediction = X * theta;
    J0 = 1/m*sum((prediction-y));
    J1 = 1/m*sum((prediction-y).*X(:,2));
    theta(1)=theta(1)-alpha*J0;
    theta(2)=theta(2)-alpha*J1;
    
 
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
