function [xtrain, ytrain, xtest, ytest] = generate_data_mnist(p_train, p_test, n_train, n_test, xpos, xneg)

[~,npos] = size(xpos); [~, nneg] = size(xneg); 
n_train_pos = floor(n_train*p_train); n_train_neg = floor(n_train-n_train_pos);
xtrainpos = xpos(:,1:n_train_pos); xtrainneg = xneg(:, 1:n_train_neg);
xtrain = [xtrainpos, xtrainneg]; ytrain = [ones(1,n_train_pos), -ones(1,n_train_neg)];

n_test_pos = floor(n_test*p_test); n_test_neg = floor(n_test - n_test_pos);
if n_train_pos + n_test_pos > npos
    display('not enough positive samples');
elseif n_train_neg + n_test_neg > nneg
    display('not enough negative samples');
else
    xtestpos = xpos(:, (n_train_pos+1):(n_train_pos + n_test_pos));
    xtestneg = xneg(:, (n_train_neg+1):(n_train_neg + n_test_neg));
    xtest = [xtestpos, xtestneg]; ytest = [ones(1,n_test_pos), -ones(1, n_test_neg)];
end