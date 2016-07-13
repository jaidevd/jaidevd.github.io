function [Weights1, Weights2, Biases1, Biases2, err] = backprop(Training, Weights1, Weights2, Biases1, Biases2, LearningRate)
% Backpropagation learning for a two layer MLP, for binary classification.
% Weights1 and Weights2 denote the weights of the first and the second
% layer, respectively. Likewise for Biases
% Training data consists of training vectos where every row is a training
% sample, with the last element of ever row as the target output.
% 
% by Jaidev Deshpande
% Roll No. 4663
% BE Elex

[m, n] = size(Training);
x = Training(:,1:(n-1));
t = Training(:,n);

% Initialize the network

% Zin = x*Weights1 + repmat(Biases1,m,1);
% Z = myactive(Zin);
% Yin = Z*Weights2 + Biases2;
% Y = myactive(Yin);

Y = testbackprop(x, Weights1, Weights2, Biases1, Biases2);

err = [];

iter = 0;

while ~isequal(Y, t)
    
    iter = iter + 1;
    
    
    for i = 1:m
        
        z_in = x(i,:)*Weights1 + Biases1;
        z = myactive(z_in);
        y_in = z*Weights2 + Biases2;
        y = myactive(y_in);
        
        delk = (t(i) - y) * y * (1-y);
        delW2 = LearningRate*delk*z';
        delB2 = LearningRate*delk;
        delin = delk*Weights2;
        delj = delin.*(myactive(z_in).*(1-myactive(z_in)))';
        delW1 = LearningRate*(delj*x(i,:))';
        delB1 = LearningRate*delj';
        
        Weights1 = Weights1 + delW1;
        Weights2 = Weights2 + delW2;
        Biases1 = Biases1 + delB1;
        Biases2 = Biases2 + delB2;
        
    end
    
    Zin = x*Weights1 + repmat(Biases1,m,1);
    Z = myactive(Zin);
    Yin = Z*Weights2 + Biases2;
    Y = myactive(Yin);

    %Y = testbackprop(x(i,:), Weights1, Weights2, Biases1, Biases2);
    
    err(iter) = sum((Y - t).^2);
    
    disp(err(iter))
    
     if err(iter)<0.0001
         break
     end
    
end

function op = myactive(ip)
op = 1./(1+exp(-ip));

function Y = testbackprop(Test, Weights1, Weights2, Biases1, Biases2)
m = size(Test, 1);
Zin = Test*Weights1 + repmat(Biases1,m,1);
Z = myactive(Zin);
Yin = Z*Weights2 + Biases2;
Y = myactive(Yin);