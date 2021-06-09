% UKF_GETSIGMA returns Sigma for the Unscented Kalman Filter
%
% Inputs:
%	nmm - (optional, struct) Neural mass model
% 
% Outputs:
%   Sigma - computed sigma according to nmm.params
function Sigma = ukf_getSigma(varargin)

% Covariances
% scale = 1e-3;
% Q1 = (scale*B)*B;
% Q2 = (scale*b)*b;
% %Q3 = sigma*sigma%(scale*mu)*mu;
% Q3 = (scale*mu)*mu
% 
% % initialise state error covariance matrix
% Pxx(:,:,1) = diag([Q1,Q2,Q3,R*ones(1,6)]);
% Sigma_scale = 1e-7;                                                         % this is our trust parameter for our model
% Sigma = diag([Q1,Q2,Q3,Sigma_scale*ones(1,6)]);

Sigma = 1e-3*eye(7); 
% Sigma([5 6 7], [5 6 7]) = 0;%Sigma([5 6 7]) .* 1e-6;
% Sigma([2 4], [2 4]) = Sigma([2 4], [2 4]) .* 10;
end