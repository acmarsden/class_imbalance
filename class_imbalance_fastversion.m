%{
clear all;
d = 10000; n_train = 10000; m = 40; p = 0.8; eps = 0; t = 10;
theta_true = -1+ 2*rand(d,1); theta_true = theta_true/norm(theta_true);

x = -1 + 2*rand(d,n_train); eps_x = eps*(rand(n_train,1) - 0.5);
y_corr = sign(theta_true'*x + eps_x');
y = sign(theta_true'*x);

% Generate test distribution with specified positive probability
z = -1 + 2*rand(d,3*m);
yz = sign(theta_true'*z);
n_pos = m*p; n_neg = m-n_pos;
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
%}
p = p_pos;
% Iterate to find an estimate of the true theta
n_iters = 10;
theta_curr_fast = -1+ 2*rand(d,1);
frac_pos = zeros(n_iters+1,1); errors_fast = Inf*ones(n_iters+1,1);
thetas = zeros(d,n_iters+1);
for rr = 1:n_iters
    thetas(:,rr) = theta_curr_fast;
    frac_pos(rr) = sum(sign(theta_curr_fast'*z)==1)/m;
    errors_fast(rr) = sum(sign(theta_curr_fast'*(yz_true.*z)) == -1)/m;
    [v,c] = compute_vc_beta(10, theta_curr_fast, z, p);
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
        errors_fast(rr+1) = sum(sign(theta_curr_fast'*(yz_true.*z)) == -1)/m;
        thetas(:,rr+1) = theta_curr_fast;
    end
end
[error_fast, best_ind] = min(errors_fast);
best_theta = thetas(:,best_ind);
display(strcat('accuracy our method, fast version: ', num2str(1-error_fast)));


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
