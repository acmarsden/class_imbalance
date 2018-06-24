clear all;
d = 20; n_train = 20; m = 30; p_pos = 0.2; eps = 0; t = 10;
theta_true = -1+ 2*rand(d,1); theta_true = theta_true/norm(theta_true);

% Generate training distribution randomly
x = -1 + 2*rand(d,n_train); eps_x = eps*(rand(n_train,1) - 0.5);
y_corr = sign(theta_true'*x + eps_x');
y = sign(theta_true'*x);
x_pos = x(:, y_corr == 1); x_neg = x(:, y_corr == -1);
[~, nx_pos] = size(x_pos); [~, nx_neg] = size(x_neg);

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
n_iterations = 20;
theta_curr = -1+ 2*rand(d,1);
error = zeros(1, n_iterations);

p_pos = 0.3;

for rr = 1:n_iterations
    theta_new = compute_new_iter(theta_curr, eps, x_pos, x_neg, z, p_pos,t);
    yziter = sign(theta_new'*z);
    theta_curr = theta_new;
    yz_est = sign(theta_curr'*z);
    error(rr) = sum(abs(yz_est - yz_true))/(2*m);

end
error_best = min(error);
display(strcat('accuracy our method: ', num2str(1-error_best)));

% Ignore the data difference and just optimize and classify
cvx_begin quiet
    variable theta_canonical(d)
    minimize norm(theta_canonical'*x - y)
cvx_end
yz_canonical = sign(theta_canonical'*z);
error_canonical = sum(abs(yz_canonical - yz_true))/(2*m);
display(strcat('accuracy ignoring class imbalance: ', num2str(1-error_canonical)));

% Match the distribution of the data
n_train2 = n_train;
nx_neg2 = nx_neg; nx_pos2 = nx_pos;
train_p_pos = nx_pos2/(n_train2);
x_pos2 = x_pos; x_neg2 = x_neg;
if train_p_pos < p_pos
    while train_p_pos < p_pos
        x_neg2 = x_neg2(:,1:nx_neg2 - 1);
        [~, nx_neg2] = size(x_neg2);
        n_train2 = n_train2 - 1;
        train_p_pos = nx_pos2/(n_train2);
    end
else 
    while train_p_pos > p_pos
        x_pos2 = x_pos2(:,1:nx_pos2 - 1);
        [~, nx_pos2] = size(x_pos2);
        n_train2 = n_train2 - 1;
        train_p_pos = nx_pos2/(n_train2);
    end
end

x_rw = [x_pos2, x_neg2];
y_rw = sign(theta_true'*x_rw);

cvx_begin quiet
    variable theta_reweight(d)
    minimize norm(theta_reweight'*x_rw - y_rw)
cvx_end
yz_reweight = sign(theta_reweight'*z);
error_reweight = sum(abs(yz_reweight - yz_true))/(2*m);
display(strcat('accuracy removing data until it matches desired positive fraction: ', num2str(1-error_reweight)));

function theta_hat_new = compute_new_iter(theta_curr, eps, x_pos, x_neg,z,p, t)
    %X_pos is dxn_pos training data vector of the positive examples
    %X_neg is dxn_neg training data vector of the negative examples
    %Y is 1xn response to X
    %Z is dxm test data vector
    %p is the proportion of positive samples in test data
    %theta_curr is dx1 vector of our current iterate
    [~,n_pos] = size(x_pos); [~, n_neg] = size(x_neg);
    [d,m] = size(z);
    theta_curr_z = t*theta_curr'*z;
    scalar = sum(sigmoid(theta_curr_z)) - m*p;
    vector = sum((exp(-theta_curr_z)./((1+exp(-theta_curr_z)).^2)).*z*t, 2);

    cvx_begin quiet
        variable theta_hat_new(d)
        minimize square(scalar + vector'*(theta_hat_new - theta_curr))
        theta_hat_new'*x_pos >= -eps*ones(1,n_pos);
        theta_hat_new'*x_neg <= eps*ones(1,n_neg);  
        norm(theta_hat_new)<=1;
    cvx_end
end


