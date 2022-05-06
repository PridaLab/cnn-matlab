clc; clear; close all;
test_path = fileparts(matlab.desktop.editor.getActiveFilename);
cnn_path = fileparts(test_path);
addpath(test_path); addpath(cnn_path); addpath(fullfile(cnn_path, 'auxiliar'))

% === uLED or Npx probes ==================================================

% -- Get data ---

% Download test data from figshare
[lfp, fs, ground_truth] = download_test_data();

% -- Detect ---

% Detect probability of ripple
conda_env = 'C:\Users\Usuario\anaconda3\envs\cnn-env';
SWR_prob = detect_ripples_cnn(lfp, fs, 'exec_env', conda_env);


% -- Plot ---

% Plot LFP example with ripples
figure('units','normalized','pos',[0.1, 0.3, 0.8, 0.4])
itimes = [496.2*fs:497.2*fs];
plot(itimes/fs, lfp(itimes,:)/500 - [1:8], 'k')
set(gca, 'ytick', []); xlabel('Time (sec)')

% Plot ripple probability
hold on
plot(itimes/fs, SWR_prob(itimes), 'b')
set(gca, 'ytick', [0 1]); xlabel('Time (sec)')

% Plot ground truth
for irip = find(ground_truth(:,1)>itimes(1)/fs & ground_truth(:,2)<itimes(end)/fs)'
    idxs_rip = ground_truth(irip,1)*fs : ground_truth(irip,2)*fs;
    plot(idxs_rip/fs, lfp(idxs_rip,:)/500 - [1:8], 'color', [.2 .8 .5])
end

% -- Threshold ---

% Get intervals
ripples = get_intervals(SWR_prob, 'LFP', lfp, 'fs', fs, 'win_size', 0.100);

% Manual curation
channel_ripple = 4;
ripples = ripple_manual_selection(lfp, ripples, channel_ripple, fs, ...
                'autosave', fullfile(test_path, 'data', 'autosave.mat'), ...
                'save_name', fullfile(test_path, 'data', 'ripples.png'));


% === Dead channels =======================================================

% Let's say channels 3 and 6 are dead
lfp_deadch = lfp;
lfp_deadch(:,[3 6]) = nan;

% Interpolate to get 8 functional channels
good_channels = [1 2 4 5 7 8];
channels_to_interpolate = [0 1 0 1 0];
lfp_interp = interpolate_channels(lfp_deadch(:,good_channels), 'n_inters', channels_to_interpolate);

% Repeat the same as before
conda_env = 'C:\Users\Usuario\anaconda3\envs\cnn-env';
SWR_prob = detect_ripples_cnn(lfp_interp, fs, 'exec_env', conda_env);
ripples = get_intervals(SWR_prob, 'LFP', lfp_interp, 'fs', fs, 'win_size', 0.100);


% === Tetrodes or linear probe ============================================

% Let's say we have only 4 channels
lfp_4channels = lfp(:, [1 3 5 8]);

% Interpolate to get 8 functional channels. We will interpolate 1 channel
% between the first two, another one between 2nd and 3rd, and two more
% interpolated channels between the last two.
channels_to_interpolate = [1 1 2];
lfp_interp = interpolate_channels(lfp_4channels, 'n_inters', channels_to_interpolate);

% Repeat the same as before
conda_env = 'C:\Users\Usuario\anaconda3\envs\cnn-env';
SWR_prob = detect_ripples_cnn(lfp_interp, fs, 'exec_env', conda_env);
ripples = get_intervals(SWR_prob, 'LFP', lfp_interp, 'fs', fs, 'win_size', 0.100);








