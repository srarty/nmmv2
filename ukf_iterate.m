% UKF_ITERATE performs one iteration of the unscented kalman filter. 
% It is called from analytic_kalman_filter_2.m, but it can be called from
% any script after each iteration of the inverted NMM.
%
% Inputs: 
%       x_hat   - a posteriori estimated state vector in time t + 1
%       P_hat   - a posteriori estimated P matrix time t + 1
%       K       - gain of the KF
%       fe      - input firing rate to excitatory neurons
%       fi      - output   " ...
%
% Outputs: 
%       x           - states at time t
%       P           - covariance matrix at time t
%       Q           - State noise covariance
%       R           - observation noise covariance?
%       y           - recording at time t
%       H           - observation function (vector)
%       NStates     - number of states (size of x)
%       f_          - (function handle) Neural Mass Model transition 
%                       function
%       scale_kf    - multiplier for the states defined by [scale_range]
%                       to scale only some states before and rescale after 
%                       the Kalman Filter
%       scale_range - vector with states to scale (see [scale_kf])
%       ALPHA_KF_LBOUND - (boolean) if true, lower bound on alpha to 0
%       ALPHA_KF_UBOUND - (integer) upper bound on alpha
%       i_method    - integration method (string) 'euler' or 'runge'
%       DO_FILTER   - not in use
%       
%
% Artemio - April 2021
function [x_hat, P_hat, K, fe, fi] = ukf_iterate(x, P, Q, R, y, H, NStates,...
      f_, scale_kf, scale_range,ALPHA_KF_LBOUND, ALPHA_KF_UBOUND, i_method, DO_FILTER)

K = zeros(7,1);

% Scale derivatives (divide by a factor)
x(scale_range) = x(scale_range)./scale_kf;
P(scale_range, scale_range) = P(scale_range, scale_range)./scale_kf;
y = y./scale_kf;

%% Define sigma points
NSigma = 2 * NStates;	% number of sigma points
% Force symmetry on P_hat
P = (P + P')/2;
% Check eigenvalues
[~,flag] = chol(P);
if flag
    % If any is negative, find the nearest Semipositve Definite matrix
    try
        [P, k]= nearestSPD(P); % Nearest SPD, Higham (1988)
    catch ME
        if strcmp('MATLAB:svd:matrixWithNaNInf', ME.identifier)
            disp(['Error during iteration: ?']);% num2str(n)]);
        end
        rethrow(ME);
    end
    if k == -1
        % Infinite loop in the nearestSPD script. No SPD matrix found
        error('Couldn''t find the nearest SPD');
    end
end
xsigma = chol(NStates*P)'; % Pxx = chol(Pxx)'*chol(Pxx) and sigmas  = sqrt(dx)sqrt(P)
Xa = x*ones(1,NSigma) + [xsigma, -xsigma];    % these are the sigma points!!

Xa_ = zeros(size(Xa));
% P_ = zeros([size(P),size(Xa,2)]);
% P_(:,:,1) = P;

% Propogate the sigma points through the system eqs
if strcmpi('euler', i_method)
    fi_vector = zeros(1,size(Xa,2));
    fe_vector = zeros(1,size(Xa,2));
    % Euler Integration
    for i = 1:size(Xa, 2) % Iterate through all sigma points % Using a parfor loop here takes 3.5 times longer :(
%         [Xa_(:,i), P_(:,:,i), fe_vector(i), fi_vector(i)] = f_(Xa(:,i), P_(:,:,i));
%         P_(:,:,i) = P_(:,:,i) + Q;
        [Xa_(:,i), ~, fe_vector(i), fi_vector(i)] = f_(Xa(:,i), P); % We don't need P_ in the UKF, it's calculated below
    end
    fe = mean(fe_vector);
    fi = mean(fi_vector);
else
    error('Invalid integration method');
end

xtilde = sum(Xa_,2)/NSigma;        % mean of the propogated sigma points
X1 = Xa_ - xtilde*ones(1,size(Xa_,2)); % substract the mean from the columns of Xa_
% Sigma = ukf_getSigma(); % Generates a suitable Sigma (added covariance)
% Pxx = X1*X1' /NSigma + Sigma; % Pxx = X1*X1' /NSigma + Sigma; Why add Sigma?
Pxx = X1*X1' /NSigma + Q; % Fixed by Yun. Q instead of Sigma
Pxx = (Pxx + Pxx')/2; % make sure we have a symetric matrix again      

%% Update step
Y=H*Xa_;
ytilde = sum(Y)/NSigma; % the mean
Y1 = Y - ytilde*ones(1,length(Y)); 
Pyy = Y1*Y1' /NSigma + R;

Pxy = X1*Y1' /NSigma;

% Calculate the Kalman gain and update
K = Pxy/Pyy;
x_hat = xtilde+K*(y-ytilde);
% Pxx = Pxx - K * Pxy'; % Error (?)
Pxx = Pxx - K * Pyy * K';
P_hat = (Pxx + Pxx')/2;

%% Force alpha-parameters bounds above zero
if ALPHA_KF_LBOUND
    % https://stackoverflow.com/questions/38736481/bounded-kalman-filter
    % Simon (2010) Kalman Filtering with state constraints: a survey of
    % linear and nonlinear algorithms
    amax = ALPHA_KF_UBOUND;
    amin = 0;
    if ALPHA_KF_UBOUND
        % Upper and lower
        x_hat(6) = min(amax, max(amin, x_hat(6)));
        x_hat(7) = min(amax, max(amin, x_hat(7)));
    else
        % Only lower bound on zero
        x_hat(6) = max(amin, x_hat(6));
        x_hat(7) = max(amin, x_hat(7));
    end
end

% Rescale derivatives back (multiply by scale factor)
x_hat(scale_range) = x_hat(scale_range).*scale_kf;
P_hat(scale_range, scale_range) = P_hat(scale_range, scale_range).*scale_kf;

end % End function