clear all;
threshold = 0.01; p_train = 0.1; p_test = 0.1; p_test_estimate = 0.1;
n_train = 300;

[Xtrain, Ytrain, Xtest, Ytest] = data_processing(threshold, p_train, p_test, n_train);

x = Xtrain; y = Ytrain; z = Xtest; yz_true = Ytest;
[d,m] = size(Xtest);

% Ignore the data difference and just optimize and classify
cvx_begin quiet
    variable theta_canonical(d)
    minimize norm(theta_canonical'*x - y)
cvx_end
yz_canonical = sign(theta_canonical'*z);
error_canonical = sum(abs(yz_canonical - yz_true))/(2*m);
display(strcat('accuracy ignoring class imbalance: ', num2str(1-error_canonical)));


% Match the distribution of the data
[~,n_train] = size(Xtrain); 
x_pos = Xtrain(:,Ytrain==1); x_neg = Xtrain(:,Ytrain == 0);
nx_pos = sum(Ytrain); nx_neg = n_train - nx_pos;
n_train2 = n_train;
nx_neg2 = nx_neg; nx_pos2 = nx_pos;
train_p_pos = nx_pos2/(n_train2);
x_pos2 = x_pos; x_neg2 = x_neg;
if train_p_pos < p_test_estimate
    while train_p_pos < p_test_estimate
        x_neg2 = x_neg2(:,1:nx_neg2 - 1);
        [~, nx_neg2] = size(x_neg2);
        n_train2 = n_train2 - 1;
        train_p_pos = nx_pos2/(n_train2);
    end
else 
    while train_p_pos > p_test_estimate
        x_pos2 = x_pos2(:,1:nx_pos2 - 1);
        [~, nx_pos2] = size(x_pos2);
        n_train2 = n_train2 - 1;
        train_p_pos = nx_pos2/(n_train2);
    end
end

x_rw = [x_pos2, x_neg2];
[~,n_pos] = size(x_pos2); [~, n_neg] = size(x_neg2);
y_rw = [ones(1, n_pos), zeros(1, n_neg)];

cvx_begin quiet
    variable theta_reweight(d)
    minimize norm(theta_reweight'*x_rw - y_rw)
cvx_end
yz_reweight = sign(theta_reweight'*z);
error_reweight = sum(abs(yz_reweight - yz_true))/(2*m);
display(strcat('accuracy removing data until it matches desired positive fraction: ', num2str(1-error_reweight)));



p = p_test_estimate;
% Iterate to find an estimate of the true theta
n_tries = 5;
n_iters = 15;
for jj = 1:n_tries
    theta_curr_fast = -1+ 2*rand(d,1);
    frac_pos = zeros(n_iters+1,1); errors_fast = Inf*ones(n_iters+1,1);
    thetas = zeros(d,n_iters+1);
    for rr = 1:n_iters
        thetas(:,rr) = theta_curr_fast;
        frac_pos(rr) = sum(sign(theta_curr_fast'*z)==1)/m;
        errors_fast(rr) = sum(sign(theta_curr_fast'*(yz_true.*z)) == -1)/m;
        [v,c] = compute_vc_beta(10, theta_curr_fast, z, p);
        if isnan(c)
            %display('c is NaN');
            break
        end
        theta_prime = sum(y.*x,2) - sign(c)*v;
        s = -c/(theta_prime'*v);
        theta_new_fast = s*theta_prime;
        theta_new_fast = theta_new_fast/norm(theta_new_fast);
        theta_curr_fast = theta_new_fast;
        if rr == n_iters
            frac_pos(rr+1) = sum(sign(theta_curr_fast'*z)==1)/m;
            errors_fast(rr+1) = sum(sign(theta_curr_fast'*(yz_true.*z)) == -1)/m;
            thetas(:,rr+1) = theta_curr_fast;
        end
    end
    [error_fast, best_ind] = min(errors_fast);
    best_errors(jj) = error_fast; best_inds(jj) = best_ind;
end
%best_theta = thetas(:,best_ind);
best_error = min(best_errors);
display(strcat('accuracy our method, fast version: ', num2str(1-best_error)));


function [v_beta, c_beta] = compute_vc_beta(t, beta, z, p)
    %z is dxm test data
    %t is parameter for sigmoid sharpness
    %p is probability z is classified positive
    %v_beta is the vector described in document
    %beta is the vector around which the taylor approx is being made
    %beta should be current estimate of true theta
    
    [~,m] = size(z);
    beta_z = t*beta'*z; 
    v_beta = sum((exp(-beta_z)./((1+exp(-beta_z)).^2)).*z*t, 2);
    c_beta = sum(sigmoid(t*beta_z)) - m*p - v_beta'*beta;
end

