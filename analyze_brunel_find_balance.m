% Code to analyze Brunel data. Spike rate of Py and Inter vs input rate
%
% Artemio - August 2021

clear


%% Load the data

params = set_parameters('brunel');
input_rates = [1:24 25:5:300];
inhibitory_rates = zeros(1,length(input_rates));
pyramidal_rates = zeros(1,length(input_rates));
for i_rates = 1:length(input_rates)
    data_file = ['C:/Users/artemios/Documents/GitHub2/mycroeeg/simulations/CUBN/iterneurons_lfp_inputRate_' num2str(input_rates(i_rates)) '.mat'];
    load(data_file);
    
    inhibitory_rates(i_rates) = mean(R_in(500:end))/params.e0;
    pyramidal_rates(i_rates) = mean(R_py(500:end))/params.e0;
end

%% Plot

fig = figure;
yyaxis left
ax_left = scatter(input_rates, pyramidal_rates * params.e0, 5, 'filled');
xlabel('Input firing rate');
ylabel('Pyramidal firing rate');

yyaxis right
ax_right = scatter(input_rates, inhibitory_rates * params.e0, 5, 'filled');
ylabel('Inhibitory firing rate');


linkaxes([ax_left.Parent ax_right.Parent],'y');