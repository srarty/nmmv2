% FREQUENCY_SPECTRUM Receives an EEG recording or simulation and returns the frequency spectrum
%
%   out = frequency_spectrum(y,type);
%
% Inputs:
%	y	- EEG recording. If it's a matrix returns 1 output per row.
%   type	- (optional) 'fft' or 'spectrogram' (default).
%   names   - (optional) cell array [size = size(y,2)] of strings 
%               containing the names of each recording in the input.
%
% Outputs:
%   out - frequency spectrum result
% 
% Artemio | 2021
function frequency_spectrum(y, varargin)
    % Ensure y is a row vector or a matrix
    if iscolumn(y), y = y'; end
    
    % If there are optional inputs, assign them accordingly.
    type = 'spectrogram'; % Default
    names = [];
    if nargin > 1 
        for i = 1 : nargin - 1
            if iscell(varargin{i}) && size(y,1) == numel(varargin{i})
                % Input is the names of the inputs
                names = varargin{i};
            elseif strcmp('spectrogram', varargin{i}) || strcmp('fft', varargin{i})
                % Input is the type
                type = varargin{i};
            else
                % Input wasn't understood
                disp(['Input: ' num2str(i + 1) ' wasn''t understood and has been ignored.']);
            end
        end
    end
    
    % Initialize vars
    dt = 0.001; % Sampling time step
    fs = 1/dt; % Sampling frequency
    y_len = size(y,1);
    switch type
        case 'spectrogram'
            figure('Name', 'Spectrogram');
            for i = 1:y_len
                % subplot(1, y_len * 3 + 1, (i-1)*3+1:(i-1)*3+3)
                subplot(1, y_len, i)
                spectrogram(y(i,:),50,40,200,1/dt, 'yaxis');
                colorbar off; % remove colorbar
                if ~isempty(names)
                    title(names{i});
                end
            end
            c = colorbar(); % Reinsert colorbar for the last plot
            c.Label.String = 'Power/frequency(dB/Hz)';
        case 'fft'
            figure('Name', 'FFT');
            for i = 1:y_len
                x = y(i, 1:end-1);
                X = fft(x,2^nextpow2(length(x)));
                fvec = linspace(-fs/2,fs/2,length(X));
            
                subplot(y_len, 1 ,i);
                plot(fvec,fftshift(abs(X)));
                title('Plot of frequency spectrum of Heart Beat data','FontSize',12);
                xlabel('f (Hz)');
                ylabel(['|Y_' num2str(i) '(f)|']);
                box off
                xlim([0 200]);
                if ~isempty(names)
                    title(names{i});
                end
            end
            
        otherwise
            error('Didn''t recognize the type: type = %s.', type);
            
    end % End switch
end % End function
            