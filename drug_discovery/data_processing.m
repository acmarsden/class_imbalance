function [Xtrain, Ytrain, Xtest, Ytest] = data_processing(threshold, p_train, p_test, n_train)
load('drug_data.mat');

X = [Xdev, Xtest, Xtrain];
Y = [Ydev', Ytest', Ytrain'];

Y = Y(Y~=0); X = X(:, Y~=0);
[~, n] = size(Y);

Xpos = X(:,Y>threshold); Ypos = Y(Y>threshold); [~,npos] = size(Ypos);
Xneg = X(:, Y<=threshold); Yneg = Y(Y<=threshold); [~,nneg] = size(Yneg);

n_train_pos = floor(n_train*p_train); n_train_neg = floor(n_train-n_train_pos);
Xtrainpos = Xpos(:,1:n_train_pos); Xtrainneg = Xneg(:, 1:n_train_neg);
Xtrain = [Xtrainpos, Xtrainneg]; Ytrain = [ones(1,n_train_pos), zeros(1,n_train_neg)];

npos_remaining = npos - n_train_pos; nneg_remaining = nneg-n_train_neg;
if npos_remaining > 0 && nneg_remaining > 0
    n_test_pos = floor(min([npos_remaining, p_test*(npos_remaining+nneg_remaining)]));
    n_test_neg = floor((n_test_pos - p_test*n_test_pos)/p_test);
    while n_test_neg > nneg_remaining
        n_test_pos = n_test_pos - 1;
        n_test_neg = floor((n_test_pos - p_test*n_test_pos)/p_test);
    end
else
    display('not enough samples');
end

Xtestpos = Xpos(:, (n_train_pos+1):(n_train_pos + n_test_pos));
Xtestneg = Xneg(:, (n_train_neg+1):(n_train_neg + n_test_neg));
Xtest = [Xtestpos, Xtestneg]; Ytest = [ones(1,n_test_pos), ones(1, n_test_neg)];