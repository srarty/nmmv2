% Code to analyze Brunel data. 
%
% Artemio - August 2021

%% Load the data
data_file = 'C:/Users/artemios/Documents/GitHub2/mycroeeg/simulations/lfp_last.mat';
load(data_file);

%% Normalize membrane potentials (0 to 1) 
V_py_normal = V_py - min(V_py); % Remove offset
V_py_normal = V_py_normal / max(V_py_normal); % Normalize

V_in_normal = V_py - min(V_py); % Remove offset
V_in_normal = V_py_normal / max(V_py_normal); % Normalize

%% Fit
ftpos = fittype( 'c/(1+exp(-a*(x+b)))+d', 'independent', 'x', 'dependent', 'y' );
opts = fitoptions( ftpos );
opts.Display = 'Off';

% opts.Lower = [0 -10 1 0];
opts.StartPoint = [1 -1 10000 -1];
% opts.Upper = [1 10 1 0];
[fitresult1, gof1] = fit( R_in', V_in_normal', ftpos, opts );

%% Plot
figure
% scatter( R_py', V_py_normal','r'); hold on
plot(fitresult1);