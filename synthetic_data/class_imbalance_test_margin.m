d = 30; m = 30; n_train = 10; p_pos_train = 0.1; p_pos = 0.7; p_pos_estimate = 0.7; 
eps = 0; t = 10; n_repeats = 1; theta_true = -1+ 2*rand(d,1); 
theta_true = theta_true/norm(theta_true);

% Generate training distribution with specified positive probability
%x = -1 + 2*rand(d,n_train); eps_x = eps*(rand(n_train,1) - 0.5);
%y_corr = sign(theta_true'*x + eps_x');
%y = sign(theta_true'*x);
%x_pos = x(:, y_corr == 1); x_neg = x(:, y_corr == -1);
%[~, nx_pos] = size(x_pos); [~, nx_neg] = size(x_neg);



x = -1 + 2*rand(d,3*n_train);eps_x = eps*(rand(3*n_train,1) - 0.5);
y_corr = sign(theta_true'*x + eps_x');
n_pos_train = n_train*p_pos_train; n_neg_train = n_train-n_pos_train;
if sum(y_corr==1) < n_pos_train
    display('dont have enough positive examples');
else
    x_pos = x(:,y_corr==1); x_pos = x_pos(:, 1:n_pos_train);
end
if sum(y_corr == -1) < n_neg_train
    display('dont have enough negative examples');
else
    x_neg = x(:, y_corr == -1); x_neg = x_neg(:, 1:n_neg_train);
end
[~, nx_pos] = size(x_pos); [~, nx_neg] = size(x_neg);
x = [x_pos, x_neg];
y = sign(theta_true'*x);

% Generate test distribution with specified positive probability
z = -1 + 2*rand(d,3*m);
yz = sign(theta_true'*z);
n_pos = m*p_pos; n_neg = m-n_pos;
if sum(yz==1) < n_pos
    display('dont have enough positive examples');
else
    z_pos = z(:,yz==1); z_pos = z_pos(:, 1:n_pos);
end
if sum(yz == -1) < n_neg
    display('dont have enough negative examples');
else
    z_neg = z(:, yz == -1); z_neg = z_neg(:, 1:n_neg);
end
z = [z_pos, z_neg];
yz_true = sign(theta_true'*z);

%{
%our method
n_iterations = 20;
error = zeros(1, n_iterations);
margin = 2; m1 = 5;
error_best = zeros(1, n_repeats);
theta_bests = zeros(d, n_repeats);
for jj = 1:n_repeats
    error = Inf*ones(1, n_iterations);
    theta_curr = -1+ 2*rand(d,1);
    for rr = 1:n_iterations
        theta_new = compute_new_iter(theta_curr, eps, x_pos, x_neg, z, p_pos,t, m1, margin);
        yziter = sign(theta_new'*z);
        theta_curr = theta_new;
        yz_est = sign(theta_curr'*z);
        error(rr) = sum(yz_est~=yz_true)/m;
        if error(rr) <= min(error)
            theta_best =  theta_curr;
        end
    end
    error_best(jj) = min(error); theta_bests(:,jj) = theta_best;
    fprintf('.');
end
[error_best, ind] = min(error_best); theta_best = theta_bests(:,ind);
yz_ours = sign(theta_best'*z);
display(strcat('accuracy our method: ', num2str(1-error_best)));
%}
% Ignore the data difference and just optimize and classify

margin = 2; ll = 1;
cvx_begin quiet
    variable theta_canonical(d)
    minimize sum(max(0, 1- y.*(theta_canonical'*x-margin))) + ll*norm(theta_canonical)
cvx_end
yz_canonical = sign(theta_canonical'*z - margin);
yx_canonical = sign(theta_canonical'*x - margin);
error_on_train = sum(yx_canonical~=y)/n_train;
error_canonical = sum(yz_canonical~=yz_true)/m;
display(strcat('accuracy ignoring class imbalance (svm soft threshold): ', num2str(1-error_canonical)));


% Match the distribution of the data
n_train2 = n_train;
nx_neg2 = nx_neg; nx_pos2 = nx_pos;
train_p_pos = nx_pos2/(n_train2);
x_pos2 = x_pos; x_neg2 = x_neg;
if train_p_pos < p_pos_estimate
    while train_p_pos < p_pos_estimate
        x_neg2 = x_neg2(:,1:nx_neg2 - 1);
        [~, nx_neg2] = size(x_neg2);
        n_train2 = n_train2 - 1;
        train_p_pos = nx_pos2/(n_train2);
    end
else 
    while train_p_pos > p_pos_estimate
        x_pos2 = x_pos2(:,1:nx_pos2 - 1);
        [~, nx_pos2] = size(x_pos2);
        n_train2 = n_train2 - 1;
        train_p_pos = nx_pos2/(n_train2);
    end
end

x_rw = [x_pos2, x_neg2];
y_rw = sign(theta_true'*x_rw);

margin = 2; ll = 1;
cvx_begin quiet
    variable theta_reweight(d)
    minimize sum(max(0, 1- y_rw.*(theta_reweight'*x_rw-margin))) + ll*norm(theta_reweight)
cvx_end
yz_reweight = sign(theta_reweight'*z - margin);
yx_reweight = sign(theta_reweight'*x_rw - margin);
error_on_train_rw = sum(yx_reweight~=y_rw)/n_train;
error_reweight = sum(yz_reweight~=yz_true)/m;
display(strcat('accuracy removing data (svm soft threshold): ', num2str(1-error_reweight)));


%Our Method: Fast Version selects the one with closest distribution match

n_repeats = 5;
n_iters = 15;
theta_curr_fast = -1+ 2*rand(d,1);thetas = zeros(d,n_iters+1);
frac_pos = zeros(n_iters+1,1); frac_pos_list = zeros(n_repeats, 1);
errors_fast = Inf*ones(n_iters+1,1); error_fast_list = zeros(n_repeats, 1); 
theta_fast_list = zeros(d, n_repeats); wcerror_list = zeros(n_repeats,1);
for tt = 1:n_repeats
    for rr = 1:n_iters
        yz_est = sign(theta_curr_fast'*z);
        thetas(:,rr) = theta_curr_fast;
        frac_pos(rr) = sum(sign(theta_curr_fast'*z)==1)/m;
        errors_fast(rr) = sum(yz_est~=yz_true)/m;
        [v,c] = compute_vc_beta(10, theta_curr_fast, z, p_pos_estimate);
        if isnan(c)
            display('c is NaN');
            break
        end
        theta_prime = sum(y.*x,2) - sign(c)*v;
        s = -c/(theta_prime'*v);
        theta_new_fast = s*theta_prime;
        theta_new_fast = theta_new_fast/norm(theta_new_fast);
        theta_curr_fast = theta_new_fast;
        if rr == n_iters
            frac_pos(rr+1) = sum(sign(theta_curr_fast'*z)==1)/m;
            errors_fast(rr+1) = sum(yz_est~=yz_true)/m;
            thetas(:,rr+1) = theta_curr_fast;
        end
        [frac_pos_best, index] = min(abs(frac_pos - p_pos_estimate));
        error_fast_list(tt) = errors_fast(index);
        frac_pos_list(tt) = frac_pos_best;
    end
    
end

[frac_pos_best, index] = min(abs(frac_pos_list - p_pos_estimate));
error = error_fast_list(index);
display(strcat('accuracy our method, fast version, choice by frac: ', num2str(1-error)));
%display(strcat('accuracy our method, worst case: ', num2str(1-worst_error)));

%display(strcat('accuracy on test set:', num2str(sum(sign(best_theta'*x)==y)/n_train)));


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


function theta_hat_new = compute_new_iter(theta_curr, eps, x_pos, x_neg,z,p, t, m1, margin)
    %X_pos is dxn_pos training data vector of the positive examples
    %X_neg is dxn_neg training data vector of the negative examples
    %Y is 1xn response to X
    %Z is dxm test data vector
    %p is the proportion of positive samples in test data
    %theta_curr is dx1 vector of our current iterate
    %m1 is the multiplying parameter for the soft svm
    [~,n_pos] = size(x_pos); [~, n_neg] = size(x_neg);
    [d,m] = size(z); x= [x_pos, x_neg]; y = [ones(1, n_pos), -ones(1,n_neg)];
    theta_curr_z = t*theta_curr'*z;
    scalar = sum(sigmoid(theta_curr_z)) - m*p;
    vector = sum((exp(-theta_curr_z)./((1+exp(-theta_curr_z)).^2)).*z*t, 2);

    cvx_begin quiet
        variable theta_hat_new(d)
        minimize square(scalar + vector'*(theta_hat_new - theta_curr)) + m1*sum(max(0, 1- y.*(theta_hat_new'*x-margin)))
        %theta_hat_new'*x_pos >= -eps*ones(1,n_pos);
        %theta_hat_new'*x_neg <= eps*ones(1,n_neg);  
        norm(theta_hat_new)<=1;
    cvx_end
end


