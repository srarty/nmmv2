% COMPUTE_PCRB_P Compute the Posterior CRB using Bergman iteration
% 
% Inputs: t - vector of time points
%         f - transition function. For a regular Kalman fiter use 
%                 @(x)(F*x), where F is the transition matrix
%         F - transition matrix function (a function that takes the
%                 current state and returns the Jacobian). For a regular
%                 Kalman fiter use @(x)F, where F is the transition matrix
%         H - observation matrix function. For linear measurement 
%                 use @(x)H, where H is observation matrix
%         Q - process covariance
%         R - measurement covariance
%         m0 - mean of prior distribution
%         P0 - covariance of prior distribution
%         M - number of Monte Carlo samples
%
% Outputs: pcrb - the posterior CRB
%
% Kelvin Layton
% Feb 2013
%
function pcrb = compute_pcrb_P_analytic(t,f,F,H,Q,R,m0,P0,M)

N = length(t);
Nstates=length(m0);

% Initialise variables
Fisher=zeros(Nstates,Nstates,N); % Fisher information matrix
Fisher(:,:,1)=P0;
pcrb=zeros(Nstates,N);

% Initalise all trajectories
%
xk=mvnrnd(m0,P0,M)';

Rinv=inv(R);



% Compute the PCRB using a Monte Carlo approximation
nps = []; % Stores the iteration numbers of failed iterations due to P being not positive semidefinite

% Progress bar
wbhandle = waitbar(0, 'Posterior Cramer-Rao Lower Bound...');
for k=2:N
    Fhat = zeros(Nstates);
    Rinvhat = zeros(Nstates);
    
    v = mvnrnd(zeros(Nstates,1),Q,M)';
    
    % Reinitialize P
    P=zeros(Nstates,Nstates,N);
    P(:,:,1)=P0;
    
    for i=1:M % parfor
        try
            % Sample the next time point for the current trajectory realisation
            [xk(:,i), P(:,:,i)] = f(xk(:,i), P(:,:,i));
            xk(:,i) = xk(:,i) + v(:,i);
            P(:,:,i) = P(:,:,i) + Q;

            % Compute the PCRB terms for the current trajectory realisation
            Fhat = Fhat + F(xk(:,i), P(:,:,i));

            Hmat = H(xk(:,i));
            Rinvhat = Rinvhat + Hmat'*Rinv*Hmat;
        catch E
            if strcmp('MATLAB:erf:notFullReal', E.identifier) ||...
                    strcmp('stats:mvncdf:BadMatrixSigma', E.identifier)
                % P matrix is not positive definite -> Remove iteration
                nps = [nps k];
                break;
            else
                % If some other error, propagate it
                rethrow(E);
            end
        end
    end
    
    Fhat=Fhat./M;
    Rinvhat=Rinvhat./M;
        
%     Fnorm(k)=norm(Fhat);
%     Feig(k) = min(eig(Fhat));
    
    % Recursively compute the Fisher information matrix
    %
    Fisher(:,:,k) = inv( inv(Fhat*Fisher(:,:,k-1)*Fhat' + Q) + Rinvhat);
    
    % Compute the PCRB at the current time
    %
    pcrb(:,k) = diag(Fisher(:,:,k));
    
    % Update progress bar
    try wbhandle = waitbar(k/N, wbhandle); catch, delete(wbhandle); error('Manually stopped'); end
end
% Delete progress bar's handle
try delete(wbhandle); catch, error('Oops!');end