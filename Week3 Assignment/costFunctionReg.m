function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



z=X*theta;
t=sigmoid(z);
er= -y' * log(t) - (1-y)' * log(1-t);


ff=sum(er);
ff=ff/m;
k=theta .^ 2;
e=k(1);
ss=sum(k);
ss=ss-e;
ss=ss*lambda;
ss=ss/2;
ss=ss/m;
J=ff+ss;


fff=(1/m) .* (X' * (t-y));

sss=(lambda/m).*(theta);

grad=fff+sss;
grad(1)=fff(1);
% =============================================================

end
