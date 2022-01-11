%% Simulate 4 state NMM and EKF estimates. Calculate the CRB
% Can be modified to more states by changing NStates. And the model
% functions, e.g. model_NM = 4 states, model_NM_6thOrder = 6 states

% TODO: Firing rates in the background activity should be set to, aroun 1
% Hz? Check the actual value, but we know that the background activity is
% some value. For exzample, when we are away from seizures, we can
% cosntrain the firing rate to keep a value.

close all
clear

%% Options -------------------------------------------------------------- %
NStates = 4; % Number of states
NInputs = 1; % Number of external inputs (u)
NParams = 2; % Number of synaptic strength parameters (alpha_ie, alpha_ei, etc...)
NAugmented = NStates + NInputs + NParams; % Total size of augmented state vector

ESTIMATE        = true;         % Run the forward model and estimate (ESTIMATE = true), or just forward (ESTIMATE = false)
PCRB            = 0;            % Compute the PCRB (false = 0, or true > 0) The number here defines the iterations for CRB
MSE             = 0;            % Compute the MSE (false = 0, or true > 0) The number here defines the iterations for MSE
REAL_DATA       = false;         % True to load Seizure activity from neurovista recordings, false to generate data with the forward model
LFP_SIMULATION  = false;        % True if data is ground truth data from the Brunel model (REAL_DATA must be 'true')
LFP_TYPE        = 'voltage';    % Source of LFP, it can be 'current' (abstract sum of currents) or 'voltage' (linear combination of Vm_Py and Cortical Input)
TRUNCATE        = -50000;%-45000;%4500;%-25000; %-50000; %10000;%-4900;%-1000;%-9700;        % If ~=0, the real data from recordings is truncated from sample 1 to 'TRUNCATE'. If negative, it keeps the last 'TRUNCATE' samples.
SCALE_DATA      = 6/50;%30/2;%6/50;%9;%1;%6/50;      % Scale Raw data to match dynamic range of the membrane potentials in our model. Multiplies 'y' by the value of SCALE_DATA, try SCALE_DATA = 0.12
INTERPOLATE     = 0;            % Upsample Raw data by interpolating <value> number of samples between each two samples. Doesn't interpolate if INTERPOLATE == {0,1}.

REMOVE_DC       = 0;            % int{1,2} Remove DC offset from observed EEG (1) or observed and simulated (2).
SMOOTH          = 0;            % Moving average on EEG to filter fast changes (numeric, window size)
ADD_NOISE       = true;         % Add noise to the forward model's states
ADD_OBSERVATION_NOISE = false;	% Add noise to the forward model's states
C_CONSTANT      = 135; %135;          % Connectivity constant in nmm_define. It is 'J' or Average number of synapses between populations. (Default = 135)

KF_TYPE         = 'unscented';  % String: 'unscented', 'extended' (default)
ANALYTIC_TYPE   = 'analytic';   % Algorithm to run: 'pip' or 'analytic'. Only makes a difference if the filter (KF_TYPE) is 'extended' or 'none'

ALPHA_KF_LBOUND  = false;       % Zero lower bound (threshold) on alpha in the Kalman Filter (boolean)
ALPHA_KF_UBOUND  = 0;%1e3;      % Upper bound on alpha in the Kalman Filter (integer, if ~=0, the upper bound is ALPHA_KF_UBOUND)
ALPHA_DECAY     = false;        % Exponential decay of alpha-params
FIX_ALPHA       = false;        % On forward modelling, Fix input and alpha parameters to initial conditions
FIX_U           = false;        % If 'true', fixes input, if 'false' it doesn't. Needs FIX_PARAMS = true
RANDOM_ALPHA    = false;        % Chose a random alpha initialization value (true), or the same initialization as the forward model (false)
MONTECARLO      = false;        % Calculae true term P6 of the covariance matrix (P) by a montecarlo (true), or analytically (false)

PLOT            = true;         % True to plot the result of the forward model and fitting.

relativator = @(x)sqrt(mean(x.^2,2)); % @(x)(max(x')-min(x'))'; % If this is different to @(x)1, it calculates the relative RMSE dividing by whatever this value is.
% ----------------------------------------------------------------------- %

% Location of the data
if ~LFP_SIMULATION
    data_file = './data/Seizure_1.mat';
%     data_file = 'C:\Users\artemios\Dropbox\University of Melbourne\Epilepsy\Adis_data.mat';
%     data_file = 'C:\Users\artemios\Dropbox\University of Melbourne\Epilepsy\Resources for meetings\adis data\Adi_data_2.mat';
else
%     data_file = 'C:/Users/artemios/Documents/GitHub2/mycroeeg/simulations/lfp_last.mat';
    data_file = 'C:/Users/artemios/Documents/GitHub2/mycroeeg/simulations/CUBN/lfp_last.mat';
end

% Initialise random number generator for repeatability
rng(0);

%% Initialization
% params = set_parameters('alpha', mu); % Set params.u from the input argument 'mu' of set_params
% params = set_parameters('alpha');       % Chose params.u from a constant value in set_params
params = set_parameters('brunel', 30);       % Chose params.u from a constant value in set_params

N = 5000;%9800; % 148262; % LFP size: 10000 (can change) % Seizure 1 size: 148262; % number of samples
if (TRUNCATE && REAL_DATA), N = TRUNCATE; end % If TRUNCATE ~=0, only take N = TRUNCATE samples of the recording or simulation
dT = params.dt;         % sampling time step (global)
dt = 1*dT;            	% integration time step
nn = fix(dT/dt);      	% (used in for loop for forward modelling) the integration time step can be small that the sampling (fix round towards zero)
t = 0:dt:(N-1)*dt;

% model structure
% ~~~~~~~~~~~~~~~~
%           u
%           |  ___
%           | /   \
%           |/    a_ie
%      |    |      |
%      v    E      I   ^
%           |      |   |  direction of info
%           |\    a_ei
%           | \    |
%           |  \__/
%           v
%
% Populations:
%   E - excitatory(pyramidal)
%   I - inhibitory
% Gains:
%   a_ei - connectivity strength from excitatory to inhibitory (alpha_ei)
%   a_ie - connectivity strength from inhibitory to excitatory (alpha_ie)
%

u = params.u;
alpha = [params.alpha_ie; params.alpha_ei]; % Parameters in the augmented state vector. Synaptic strengths[402; 186.4];%

% Initialise trajectory state
x0 = zeros(NAugmented,1); % initial state
% x0(1:NStates) = mvnrnd(x0(1:NStates),10^1*eye(NStates)); % Random inital state
x0 = params.v0*ones(size(x0));% x0([2 4]) = 0;
x0(NStates+1:end) = [u; alpha];

% Initialize covariance matrix
P0 = 1e2*eye(NAugmented);
% Make P0 different for z-values
% P0([2 4],[2 4]) = P0([2 4],[2 4]) * 50;
P = zeros(NAugmented, NAugmented, N);
P(:,:,1) = P0;

% Initialize vector to store firing rates (output of the sigmoid)
f_i = zeros(1,N); % Firing rate of the inhibitory neurons
f_e = zeros(1,N); % Firing rate of the excitatory neurons


% Define the model
nmm = nmm_define(x0, P0, params, C_CONSTANT);
% Pass options to model struct
nmm.options.P6_montecarlo   = MONTECARLO;
nmm.options.ALPHA_DECAY     = ALPHA_DECAY;
nmm.options.ALPHA_KF_UBOUND = ALPHA_KF_UBOUND;
nmm.options.ALPHA_KF_LBOUND = ALPHA_KF_LBOUND;
nmm.options.KF_TYPE         = KF_TYPE;
nmm.options.ANALYTIC_TYPE   = ANALYTIC_TYPE;


% Initialize states
x0 = nmm.x0;
x = zeros(NAugmented,N);
x(:,1) = x0;

% Transition model
f = @(x)nmm_run(nmm, x, [], 'transition');
F = @(x)nmm_run(nmm, x, [], 'jacobian');
% Analytic
if strcmp('unscented', KF_TYPE)
    f_ = @(x,P)nmm_run(nmm, x, P,  'transition');
else
    f_ = @(x,P)nmm_run(nmm, x, P,  ANALYTIC_TYPE);
end

F_ = @(x,P)nmm_run(nmm, x, P,  'jacobian'); % F_ - transition matrix function (a function that takes the current state and returns the Jacobian).

%% Generate trajectory
% Euler integration
for n=1:N-1
    x(:,n+1) = f(x(:,n)); % Zero covariance
    if FIX_ALPHA, x(NStates+2:end,n+1) = x0(NStates+2:end); end % Fix the parameters (u and alpha)
end

% Calculate noise covariance based on trajectory variance over time??
%   Why noise on all states?
warning('Initialization of Q differs for Real data vs Simulated data');
if REAL_DATA
    Q = 1e-2*eye(NAugmented);
else
    Q = 10^-1.*diag((0.4*std(x,[],2)*nmm.params.scale*sqrt(dt)).^2); % The alpha drift increases with a large covariance noise (Q)
end
%  Q(NStates+1 : end, NStates+1 : end) =  10e-3*eye(NAugmented - NStates); % 10e-1*ones(NAugmented - NStates);

v = mvnrnd(zeros(NAugmented,1),Q,N)';% 10e-1.*mvnrnd(zeros(NAugmented,1),Q,N)';

% Get alphas from estimation
estimation = load('gt');%load('estimation_ukf'); % Load estimation results from real data (Seizure 1)
wbhandle = waitbar(0, 'Generating trajectory...'); % Loading bar
% Generate trajectory again with added noise
% Euler-Maruyama integration
for n=1:N-1
    try
        [x(:,n+1), ~, f_i(n+1), f_e(n+1)] = f(x(:,n)); % Propagate mean

        if (FIX_ALPHA), x(NStates+2:end,n+1) = x0(NStates+2:end); end % Fixing the parameters, alternative, try this (next line): 
%         if (FIX_ALPHA), x(NStates+1+~FIX_U:end,n+1)= estimation.x(NStates+1+~FIX_U:end, min(n+1,size(estimation.x, 2))); end % Fixing the parameters to the result of a previous recording.
        x(:,n+1) = x(:,n+1) + (ADD_NOISE * v(:,n)); % Add noise if the ADD_NOISE option is true.
    catch ME % Try catch around the whole for loop to make sure we close the progress bar in case there is an error during execution.
        if exist('wbhandle','var')
            delete(wbhandle)
        end
    end
    % Update progress bar
    try wbhandle = waitbar(n/(N-1), wbhandle); catch, delete(wbhandle); error('Manually stopped: Forward model.'); end
end
% Remove progress bar
delete(wbhandle);

% Observation function (H = [1 0 0 0 1 0 0]) <- state x1 and parameter u
% are in the observation matrix.
H = zeros(1, NAugmented);
H(1) = 1;        
H(NStates + 1) = 1;
H = H.*1; % Scale the observation matrix if needed
% Scale
H = H/nmm.params.scale;

R = 1e-1*eye(1);

w = ADD_OBSERVATION_NOISE .* mvnrnd(zeros(size(H,1),1),R,abs(N))'; % Absolute value of N accounts for when TRUNCATE is negative (truncating the initial part of the recording)
y = H*x + w;

if REMOVE_DC == 2
    % Remove DC offset from simulation (Observed EEG)
    y = y - mean(y(length(y)/2:end));
%     y = y - min(y(length(y)/2:end));
end

if SMOOTH
    y = smooth(y,SMOOTH);
end

if ~ESTIMATE
    %% Plot only the forward model results (generated trajectory)
    % Plot x(1) and ECoG
    figure
    ax1=subplot(211);
    plot(t,x(1,:)');
    ylabel('State 1');
    ax2=subplot(212);
    plot(t,y);
    ylabel('ECoG (mV)');
    xlabel('Time (s)');
    linkaxes([ax1 ax2],'x');

    % Plot all 4 states
    figure
    axs = nan(NStates,1); % Axes handles to link subplots x-axis
    for i = 1:NStates
        axs(i) = subplot(NStates, 1, i);
        plot(t(1:min(length(t),size(x,2))),x(i,1:min(length(t),size(x,2)))');
        ylabel(['State ' num2str(i)]);
    end
    linkaxes(axs, 'x');
    xlabel('time');
    
    % Plot external input and augmented parameters
    figure
    axs = nan(NAugmented - NStates,1); % Axes handles to link subplots x-axis
    for i = 1:NAugmented - NStates
        axs(i) = subplot(NAugmented - NStates, 1, i);
        plot(t(1:min(length(t),size(x,2))),x(NStates + i,1:min(length(t),size(x,2)))');
        ylabel(['Parameter ' num2str(i)]);
    end
    linkaxes(axs, 'x');
    xlabel('Time (s)');
    
    % Firing rates (Output of the nonlinearity)
    figure
    ax1 = subplot(2,1,1);
    yyaxis left
    plot(t, f_e);
    title('Sigmoid function output');
    ylabel('f_e');
    yyaxis right
    plot(t, f_e * params.e0);
    
    ax2 = subplot(2,1,2);
    yyaxis left
    plot(t, f_i);
    ylabel('f_i');
    linkaxes([ax1 ax2],'x');
    xlabel('Time (s)');
    yyaxis right
    plot(t, f_i * params.e0);
    
    return
end
    
if REAL_DATA
    % Load real data from .mat :
    load(data_file);  % change this path to load alternative data
    if ~LFP_SIMULATION
        % Real iEEG recordings (neurovista)
        Ch = 1; % Channel
        y = Seizure(:,Ch)';
%         y = Seizure(:,Ch)' + 50;
%         y = norm_lfp; 
%         y = lfp;
    else
        % Ground truth
        if strcmp('current', LFP_TYPE), Seizure = LFP; else, Seizure = LFP_V; end
        y = Seizure; % Load the data
        y = reshape(y,1,length(y));% Ensure it's horizontal
        x = zeros([size(x,1) size(Seizure,2)]); 
        x(1,:) = (V_py - v_rest) * 1e3; % Substract the resting membrane potential from the Brunel and scale to remove mV
        x(2,:) = [0 diff(x(1,:))/(lfp_dt)];
        x(3,:) = (V_in - v_rest) * 1e3;
        x(4,:) = [0 diff(x(3,:))/lfp_dt];
        % u could be 1 sample longer than x:
        try
            x(5,:) = u;
        catch E
            x(5,:) = u(2:end);
        end
        if numel(alpha1) == 1
            % this parameter can be size 1 or longer, if it's a vector it means
            x(6,:) = alpha1 * (nmm.x0(6)/nmm.params.alpha_ie); % alpha1 is the parameter from Brunel. Divide by all other stuff to complete the lumped parameter as NMM
        else
            x(6,:) = interp(alpha1,0.2/dt); % dt of the monkey's LFP is 0.02 s
        end
        x(7,:) = abs(alpha2) * (nmm.x0(7)/nmm.params.alpha_ei);
        
%         x0 = x(:,1);
    end
    
%     
%     if SCALE_DATA %#ok<BDLGI> % Removes the warning for SCALE_DATA being constant
%         y = y * SCALE_DATA; %0.12;
%     end   
        
    if REMOVE_DC ~= 0
        % Remove DC offset from real or ground truth data
        y = y - mean(y(length(y)/2:end));
%         y = y + 30;
%         y = y - min(y(length(y)/2:end)) + 10; % Adjusting to the range 40 - 100 from the forward NMM
        y = y - 145; %16.25;
    end
    
    % Check if the data contains a time stamp
    if ~exist('T', 'var')
        if ~LFP_SIMULATION
%             T = 4.0694e6; % Hardcoded data taken from Seizure_1.mat
            T = length(y) / fs;
        else
            T = lfp_dt * length(y);
        end
    end
    % Define the time step
    params.dt = T/length(y); % 1e-3*T/length(y); % 1e-3 because we want miliseonds
    nmm.params.dt = params.dt;
        
    if TRUNCATE > 0
        % Truncate from the beginning
        y = y(1:N);
        if LFP_SIMULATION
            x = x(:,1:N);
        end
    elseif TRUNCATE < 0
        % Truncate from end
        y = y(end+N+1 : end);
        if LFP_SIMULATION
            x = x(:,end+N+1 : end);
        end
    end
    
    if SCALE_DATA %#ok<BDLGI> % Removes the warning for SCALE_DATA being constant
        y = y * SCALE_DATA; %0.12;
        warning('Hardcoding DC offset');
        y = y + 30;%- 150;%30;
    end    
end

if INTERPOLATE
    y_ = y; % Change of variable to keep the original recording
    y = interp(y_,INTERPOLATE); % Upsample raw data with interpolated values
    t_ = params.dt:params.dt:params.dt*length(y_); % Store original time series
    t = (params.dt:params.dt:params.dt*length(y))./INTERPOLATE; % Upsampled time series
    params.dt = params.dt/INTERPOLATE;
    if (~REAL_DATA || LFP_SIMULATION)
        % If not real data, i.e. simulated data, then interpolate the x as
        % well
        x_ = x; % Store original state vector
        x = zeros(size(x,1), size(x,2) * INTERPOLATE);
        for i = 1:size(x,1)
            x(i,:) = interp(x_(i,:),INTERPOLATE); % Upsample raw data with interpolated values
        end
    end
else
    t_ = params.dt:params.dt:params.dt*length(y);
    t = t_;
end

%% Run KF for this model
% Prior distribution (defined by m0 & P0)
% m0 = params.v0*ones(size(x0));% m0([2 4]) = 0; 
m0 = mean(x(:,ceil(size(x,2)/2):end),2); %
m0(5) = x0(5) + RANDOM_ALPHA * (x0(5)*rand());%0;% mean(y(ceil(size(y,2)/2):end)); % x0(5);%32;%
m0(6) = x0(6) + RANDOM_ALPHA * (x0(6)*(rand()-0.05));
m0(7) = x0(7) + RANDOM_ALPHA * (x0(7)*(rand()-0.05));
nmm.x0 = m0; % Update initial value in nmm, i.e. nmm.x0

% P0 = 1e2*eye(NAugmented); % P0 will use the same initial value as the
% forward model
% P0(NAugmented - NParams + 1 : end, NAugmented - NParams + 1 : end) = 1e3*eye(NParams);

% Calculate P0 from the forward simulation
P0 = cov(x(:,ceil(size(x,2)/2):end)');
P0(5,5) = 100;%100
P0(6,6) = 10000;%1e6; 1000
P0(7,7) = 10000;%1e3; 100
nmm.P0 = P0;

% Apply KF filter chosen in options at the top
try
    tic
    [m, Phat, ~, fi_exp, fe_exp] = analytic_kalman_filter_2(y,f_,nmm,H,Q,R,'euler');
    toc_ = toc;
    disp(['Kalman filter estimation took: ' num2str(toc_) ' seconds']);
catch ME
    if strcmp('Manually stopped', ME.message)
        disp('Kalman filter manually stopped by user.');
        return
    else
        rethrow(ME);
    end
end

% y_ekf = H*m_;% + w;
y_analytic = H*m;% + w;

%% Plot results
if PLOT
    if (~REAL_DATA || LFP_SIMULATION)
        %% Plot x(1) and ECoG
        figure
        ax1=subplot(211);
        plot(t,x(1,:)'); hold on;
        plot(t,m(1,:)','--'); % EKF
    %     plot(t,m_(1,:)','--'); % Analytic, rung kuta
        % plot(t,m__([1],:)','--'); % Analytic euler
        legend({'Actual','Estimation'});
        ylabel('V_{Py}');
        ax2=subplot(212);
        plot(t,y); hold on;
    %     plot(t,y_ekf, '--');
        plot(t,y_analytic, '--');
        legend({'Observed EEG', 'Estimated EEG'});
        % plot(t,pcrb(1,:)')
        % legend({'CRLB'})
        ylabel('ECoG (mV)');
        xlabel('Time (s)');
        linkaxes([ax1 ax2],'x');

        %% Plot all 4 states
        figure
        axs = nan(NStates,1); % Axes handles to link subplots x-axis
        labels_y = {'V_{Py}' 'z_{Py}' 'V_{I}' 'z_{I}' 'u' '\alpha{}_1' '\alpha{}_2'};
        for i = 1:NStates
            axs(i) = subplot(NStates, 1, i);
            %plot(t_(1:min(length(t_),size(x,2))),x(i,1:min(length(t_),size(x,2)))'); hold on;
            plot(t,x(i,:)'); hold on;            
            plot(t,m(i,:)','--');
%             plot(t,m_(i,:)','--');
%             plot(t,m__(i,:)','--');
%             ylabel(['State ' num2str(i) '(' labels_y{i} ')']);
            ylabel(labels_y{i});
        end
        linkaxes(axs, 'x');
        legend({'Simulation', 'Estimation'});
        xlabel('time');

        %% Plot external input and augmented parameters
        figure
        axs = nan(NAugmented - NStates,1); % Axes handles to link subplots x-axis
        for i = 1:NAugmented - NStates
            axs(i) = subplot(NAugmented - NStates, 1, i);
            plot(t, x(NStates + i,:)'); hold on;
            plot(t, m(NStates + i,:)','--');
            plot(t, zeros(size(t)), '--', 'Color', [0.8 0.8 0.8]);
            ylabel(labels_y{i + NStates});
        end
        linkaxes(axs, 'x');
        xlabel('time');
        legend({'Simulation', 'Estimation'});

        %% Firing rates (Output of the nonlinearity)
        figure
        ax1 = subplot(2,1,1);
        if ~LFP_SIMULATION
            plot(t_, f_e);        
            hold
        end
        plot(t, fe_exp, '--');
        title('Sigmoid function output');
        ylabel('f_e');
        ax2 = subplot(2,1,2);
        if ~LFP_SIMULATION
            plot(t_, f_i);
            hold
        end
        plot(t,fi_exp, '--');
        ylabel('f_i');
        linkaxes([ax1 ax2],'x');
        xlabel('Time (s)');

        %% Covariance (Estimation)
        figure
        % plt = @(x,varargin)plot(t,squeeze(x)./max(abs(squeeze(x))),varargin{1});
        plt = @(x,varargin)plot(t,squeeze(x),varargin{1}); % Estimation
        plt_ = @(x,varargin)plot(t_,squeeze(x),varargin{1}); % Forward (time vector is different)
        for i = 1:NAugmented
            subplot(2,4,i)
            % Forward model
%             plt_(P(i,i,:),'-'); hold on;
            % Estimation
            plt(Phat(i,i,:),'--');hold on;
    %         plt(Phat_(i,i,:),'--');
        end
        %legend({'Simulation', 'Analytic KF'});
        subplot(2,4,1);
        title('Covariance matrix (P) - Diagonal');

%         %% Covariance of alpha vectors
%         figure
%         for i = 1:NAugmented
%             for j = 1:NAugmented
%                 subplot(NAugmented,NAugmented,i + NAugmented*(j-1));
%                 % Estimation
%                 plt(Phat(j,i,:),''); hold on;
%                 xlim([0.2 5]);
%     %             ylim([-100000 100000]);
%             end
%         end

    else
        %%
        % If estimating real data
        figure
        ax1=subplot(211);
        plot(t,m([1],:)','-'); hold on; % Analytic KF
        legend({'Prediction'});
        ylabel('State 1 (Vm)');

        ax2=subplot(212);
        plot(t,y_analytic, '--', 'LineWidth', 2);hold on
        plot(t,y);
        linkaxes([ax1 ax2],'x');
        ylabel('ECoG');
        xlabel('Time (s)');
        % Here, find and plot places where alpha's cross through zero
        zci = @(v) find(diff(sign(v))); % Zero cross detector
        try plot(t(zci(m(6,:))), max(y) + 1,'v', 'Color', [0.4 0.6 0.9], 'MarkerSize', 8, 'LineWidth', 2); catch, end
        try plot(t(zci(m(7,:))), max(y) + 1,'v', 'Color', [0.8 0.4 0.1], 'MarkerSize', 8, 'LineWidth', 2); catch, end
        % Rapid change in alpha
        hdi = @(v) find(diff(abs(v)) > 30 * (mean(abs(diff(v))) + std(abs(diff(v)))) ~= 0);
        try plot(t(hdi(m(6,:))), -(max(y) + 1),'^', 'Color', [0.4 0.6 0.9], 'MarkerSize', 8, 'LineWidth', 2); catch, end
        try plot(t(hdi(m(7,:))), -(max(y) + 1),'^', 'Color', [0.8 0.4 0.1], 'MarkerSize', 8, 'LineWidth', 2); catch, end

        legend('Prediction','Observed ECoG'); % legend('EKF', 'Analytic (euler)','Observed EEG');

        % Plot the 4 states
        figure
        axs = nan(NStates,1); % Axes handles to link subplots x-axis
        for i = 1:NStates
            axs(i) = subplot(NStates, 1, i);
            plot(t,m(i,:)','-');hold on;
            ylabel(['State ' num2str(i)]);
        end
        linkaxes(axs, 'x');
        xlabel('Time (s)');

        % Plot the 3 parameters
        figure
        axs = nan(NAugmented-NStates,1); % Axes handles to link subplots x-axis
        for i = NStates+1:NAugmented
            axs(i-NStates) = subplot(NAugmented-NStates, 1, i-NStates);
            plot(t,m(i,:)','-');hold on;
            plot(t, zeros(size(t)), '--', 'Color', [0.8 0.8 0.8]);
            ylabel(['Parameter ' num2str(i-NStates)]);
        end
        linkaxes(axs, 'x');
        xlabel('Time (s)');

        % Firing rates (Output of the nonlinearity)
        figure
        ax1 = subplot(2,1,1);
        plot(t, fe_exp, '-');
        title('Sigmoid function output (prediction)');
        ylabel('f_e');
        ax2 = subplot(2,1,2);
        plot(t,fi_exp, '-');
        ylabel('f_i');
        linkaxes([ax1 ax2],'x');
        xlabel('Time (s)');

        % Covariance (Estimation)
        figure
        % plt = @(x,varargin)plot(t,squeeze(x)./max(abs(squeeze(x))),varargin{1});
        plt = @(x,varargin)plot(t,squeeze(x),varargin{1});
        plt_ = @(x,varargin)plot(t,squeeze(x),varargin{1});
        for i = 1:NAugmented
            subplot(2,4,i)
            % Estimation
            plt(Phat(i,i,:),''); hold on;
            % Forward model
    %         plt_(Phat_(i,i,:),'--');
        end
    %     legend({'EKF' 'Analytic'});
        legend({'Analytic'});
        subplot(2,4,1);title('Covariance matrix (P) - Diagonal'); 

        % Covariance of alpha vectors
%         figure
%         for i = 1:NAugmented
%             for j = 1:NAugmented
%                 subplot(NAugmented,NAugmented,i + NAugmented*(j-1));
%                 % Estimation
%                 plt(Phat(j,i,:),''); hold on;
%             end
%         end
    end
    % Nice placement of figures
    poss = [104, 562, 560, 420; 694, 563, 560, 420; 1288, 562, 560, 420; 107,  49, 560, 420; 693,  51, 560, 420; 1287,  50, 560, 420];
    for i = 1:5, figs{i}=figure(i); end
    for i = 1:5, figs{i}.Position = poss(i,:); end
end % If PLOT

%% Compute the posterior Cramer-Rao bound (PCRB)
if ~PCRB && ~MSE
    return
elseif PCRB
    M = PCRB;    % Number of Monte Carlo samples % PCRB contains the number of iterations on the Cramer-Rao bound's calculation and the MSE
    
    try
        pcrb_analytic = sqrt(compute_pcrb_P_analytic(t,f_,F_,H,Q,R,m0,P0,M,y, ALPHA_KF_LBOUND, ALPHA_KF_UBOUND, KF_TYPE)) ./ relativator(x); % Divided by the range of the data to calculate the relative rmse % Square root to compare it to the Root Mean Square Error
        % pcrb = sqrt(compute_pcrb_P(t,f,F,@(x)H,Q,R,m0,P0,M)); % Square root to compare it to the Root Mean Square Error
        % pcrb = compute_pcrb_P(t,f_,F,@(x)H,Q,R,m0,P0,M); % f_ for analytic KF
        % pcrbx5 = compute_pcrb_P(t,f,F,@(x)H,Q,R,m0,P0.*5,M); % Changed initial condition, multiply P0 by 5
        % pcrbd5 = compute_pcrb_P(t,f,F,@(x)H,Q,R,m0,P0./5,M); % Changed initial condition, divide P0 by 5
    catch E
        if strcmp('Manually stopped', E.message)
            disp('PCRB manually stopped by user.');
            return
        else
            rethrow(E);
        end
    end
end

%% Compute the MSE of the extended Kalman filter
num_trials = MSE; % PCRB contains the number of iterations on the Cramer-Rao bound's calculation and the MSE
if ~REAL_DATA
    err = zeros(NAugmented,N * max(1,INTERPOLATE));
    error_ = zeros(NAugmented,N * max(1,INTERPOLATE));
    % error__ = zeros(NAugmented,N);
else
    err = zeros(size(y));
    error_ = zeros(size(y));
end
nps = 0; % Non-positive semidefinite P matrix, iteration counter for removal
% parfor r=1:num_trials
% To avoid calculating the new trajectory every iteration. Comparing to the "real" value generated above
z = y; 
% Progress bar
wbhandle = waitbar(0, 'Calculating MSE...');
for r=1:num_trials
    
%     % Create new trajectory realisation
%     %
%     v = mvnrnd(zeros(1,NAugmented)',Q,N)';
%     x = zeros(NAugmented,N);
%     x(:,1)=mvnrnd(m0,P0)';
%     for i=NAugmented:N
%         x(:,i) = f(x(:,i-1)) + v(:,i);
%     end
% 
%     % Generate new observations 
%     %
%     w = mvnrnd(zeros(1,size(H,1)),R,N)';
%     z = H*x + w;
    
    % Apply EKF filter
%     m = extended_kalman_filter_2(z,f,F,H,Q,R,m0,P0);
    try
         m_ = analytic_kalman_filter_2(z,f_,nmm,H,Q,R,'euler',1,false,true);
%         m__ = analytic_kalman_filter_2(z,f_,F_,H,Q,R,m0,P0,'runge');
    catch E
        if strcmp('MATLAB:erf:notFullReal', E.identifier) ||...
                strcmp('stats:mvncdf:BadMatrixSigma', E.identifier)
            % P matrix is not positive definite -> Remove iteration
            nps = nps + 1; % Fails counter
            % Error is 0 for failed iterations, nps is subtracted from the
            % total number of iterations to calculate MSE.
            % Continue with next iteration of the for loop without adding
            % any error. This iteration won't affect the MSE.
            continue;
        elseif strcmp('Couldn''t find the nearest SPD', E.message)
            disp(['Error found while Running MSE , iteration: ' num2str(r)]);
            % P matrix is not positive definite -> Remove iteration
            nps = nps + 1; % Fails counter
            % Error is 0 for failed iterations, nps is subtracted from the
            % total number of iterations to calculate MSE.
            % Continue with next iteration of the for loop without adding
            % any error. This iteration won't affect the MSE.
            continue; 
        else
            % If some other error, propagate it
            rethrow(E);
        end
    end

    if ~REAL_DATA
        % Accumulate the estimation error
%         error = error + (x-m).^2;
        error_ = error_ + (x-m_).^2;
    %     error__ = error_ + (x-m__).^2;
    else
%         y_analytic = H*m;% + w;
        y_ekf = H*m_;% + w;
        
        error_ = error_ + (y-y_ekf).^2;
%         error = error + (y-y_analytic).^2;
    end
    % Update progress
    try wbhandle = waitbar(r/num_trials, wbhandle); catch, delete(wbhandle); error('MSE manually stopped by user.'); end
end
try delete(wbhandle); catch, error('Oops!');end

% Calculate the mean squared error
num_trials = num_trials - nps; % Subtract failed iterations
% Check how many nps were subptracted, i.e. failed runs of the MSE
if num_trials <= 0
    error('Not enough successful iterations when calculating MSE');
end

if ~REAL_DATA
%     rmse = sqrt(error ./ num_trials) ./ relativator(x); % Divided by the range of the data to calculate the relative rmse
    rmse_ = sqrt(error_ ./ num_trials) ./ relativator(x);
    % mse__ = error__ ./ num_trials./relativator(x);
    % rmse = sqrt(error ./ num_trials);
else
    rmse_ = sqrt(error_ ./ num_trials) ./ relativator(y); % Divided by the range of the data to calculate the relative rmse
end

%% Plot MSE and the PCRB vs Time
if PCRB && MSE % Both
    figure('Name', 'NMM - EKF vs CRB')
    if ~REAL_DATA
        for i = 1:NAugmented
            subplot(2,4,i)
            semilogy(t,rmse_(i,:),'.-'); hold on;
            semilogy(t,pcrb_analytic(i,:),'.-');
            grid on;
            xlabel('Time (s)');
            ylabel(['RMSE state ' num2str(i)]);
            hold off;
        end
    else
        for i = 1:NAugmented
            subplot(2,4,i);
            semilogy(pcrb_analytic(i,:),'.-'); hold on;
            grid on;
            xlabel('Time (s)');
            ylabel(['RMSE state ' num2str(i)]);
            legend({'PCRB', 'PCRB Analytic'});
            ylim([10^-1 10^5]);
        end
        
        figure
        semilogy(t,rmse_,'.-'); hold on;
        ylim([10^-6 10^6]);
        grid on;
    end
    % legend({'RMSE (EKF)', 'RMSE (Analytic - Euler)', 'RMSE (RK)', 'PCRB', 'PCRB Analytic'});
    legend({'RMSE', 'PCRB'});
    
% Only MSE    
elseif MSE
    figure(7)%, 'Name', 'Mean Square Error over time')
    if ~REAL_DATA
        for i = 1:NAugmented
            subplot(2,4,i)
            semilogy(t,rmse_(i,:),'.-');
            grid on;
            xlabel('Time (s)');
            ylabel(['RMSE state ' num2str(i)]);
        end
    else
%         semilogy(t,rmse_,'.-');
%         plot(t,movmean(rmse_,[100 100]));
        semilogy(t,movmean(rmse_,[100 100]));
        hold on
%         ylim([10^-6 10^6]);
        grid on;
    end
    % legend({'RMSE (EKF)', 'RMSE (Analytic - Euler)', 'RMSE (RK)', 'PCRB', 'PCRB Analytic'});
    legend({'RMSE'});
end

%% Plot average RMSE vs CRB 
figure('Name', 'Mean Vm - EKF vs CRB')
color = [0.1,0.6,0.7];
hAx = plot(mean(rmse_,2),'o',...
    'MarkerSize',10,...
    'MarkerEdgeColor','k',...
    'MarkerFaceColor',color);
hold on;

if PCRB
    bar(mean(pcrb_analytic,2), 'FaceColor', color+0.1);
%     bar(mean(pcrb,2), 'FaceColor', color-0.1,'BarWidth', 0.4);
end

% semilogy(mean(rmse_,2),'o',...
%     'MarkerSize',10,...
%     'MarkerEdgeColor','k',...
%     'MarkerFaceColor',color+0.2);
% semilogy(mean(mse__,2),'^',...
%     'MarkerSize',10,...
%     'MarkerEdgeColor','k',...
%     'MarkerFaceColor',color+0.2);
yt = get(gca, 'YTick'); YTickLabels = cellstr(num2str(round(log10(yt(:))), '10^%d')); % Format y axis in log scale
grid on;
if PCRB
    legend({'RMSE (Analytic)', 'PCRB Analytic','PCRB','RMSE (Euler)', 'RMSE (RK)'});
else
    legend({'RMSE (EKF)', 'RMSE (Euler)'});
end    
uistack(hAx,'top');
xlabel('Time (s)');
ylabel('RMSE (%)');
% ylim([10^-3 10^0])

%% Plot the evolution of the CRB
if PCRB
    figure('Name', 'CRB Convergence')
    for i = 1:NAugmented
        StateNum = i;

%         pcrb_mvnavg = movmean(pcrb,1,2); % Sqrt of PCRB original initial conditions
        pcrb_analytic_mvnavg = movmean(pcrb_analytic,1,2); % Sqrt of PCRB original initial conditions
        % pcrbx5_sqrt = movmean(sqrt(pcrbx5),1,2); % Sqrt of PCRB when P0 is 5 times bigger
        % pcrbd5_sqrt = movmean(sqrt(pcrbd5),1,2); % Sqrt of PCRB when P0 is 5 times smaller
        range_ = 10:length(t)/2; % Range of values to plot to avoid the initial overshoot and to remove the entries beyond convergence

        subplot(2,4,i)
%         plot(t(range_), pcrb_mvnavg(StateNum, range_), 'LineWidth', 2);
%         hold on;
        plot(t(range_), pcrb_analytic_mvnavg(StateNum, range_), 'LineWidth', 2);
        % plot(t(range_), pcrbx5_sqrt(StateNum, range_), '--', 'LineWidth', 2);
        % plot(t(range_), pcrbd5_sqrt(StateNum, range_), '--', 'LineWidth', 2);
        % legend({'P0 = 10000I', 'P0 = 50000I', 'P0 = 2000I'});
        grid on;
        title(['State: ', num2str(StateNum)]);
        xlabel('Time (s)');
        ylabel('BCRB(mV)');
        % xlim([0 5]);
    end
    legend('PCRB', 'PCRB (Analytic)');
end
%%
if PCRB || MSE
    figure
%     subplot(2,1,1)
    if ~REAL_DATA
        semilogy(t,sum(rmse_),'x-')
    else
        semilogy(t,rmse_,'x-')
    end
    if PCRB
        hold on;
        % semilogy(t,sum(mse__),'x-')
        semilogy(t,sum(pcrb_analytic),'o-');
    end
    grid on;
    xlabel('Time (s)');
    ylabel('RMSE');
    legend('RMSE', 'PCRB');
    
%     subplot(2,1,2)
%     if ~REAL_DATA
%         semilogy(t,sum(rmse_),'x-'); hold on;
%     else
%         semilogy(t,rmse_,'x-'); hold on;
%     end
%     semilogy(t,sum(pcrb_analytic),'o-');
%     legend('RMSE Analytic', 'PCRB Analytic');
%     grid on;
%     xlabel('Time (s)');
%     ylabel('RMSE');
%     legend('RMSE', 'PCRB');
end