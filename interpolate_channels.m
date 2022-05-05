function y_interp = interpolate_channels(y, varargin)

%%%  Adds interpolated channels between any pair
%%%
%%%  >> y_interp = interpolate_channels(y, varargin)
%%%
%%% Inputs:
%%%    y            n_samples x n_channels LFP matrix
%%%    n_inters     (optional) Number of interpolations between each 
%%%                 channel pair. It needs to be of (n_channels-1) length.
%%%                 By default it will add one interpolated channel between
%%%                 each pair.
%%%
%%% Output:
%%%    y_interp     LFP matrix with interpolated channels
%%% 
%%% 
%%% Examples:
%%%
%%%     Let's say size(y) = [N, 4], but we want 8 channels, adding one
%%%     channel between each pair, except between the first two channels,
%%%     between which we want 2 interpolated channels. Then:
%%%          >> y_interp = interpolate_channels(y, 'n_inters', [2 1 1])
%%%
%%%     Let's say now that size(y) = [N, 8], but channel 6 is dead. Then we
%%%     could do:
%%%          >> y_interp = interpolate_channels(y(:, [1 2 3 4 5 7 8]), ...
%%%                             'n_inters', [0 0 0 0 1 0])
%%% 
%%% A. Navas-Olive, LCN 2022

    p = inputParser;
    addParameter(p,'n_inters', [], @isnumeric);
    parse(p,varargin{:});
    n_inters = p.Results.n_inters;
    
    % Number of channels
    n_ch = size(y,2);
    
    % Number of interpolations between each channel pair
    if isempty(n_inters)
        n_inters = ones(1,n_ch-1);
    end
    
    % Initialize
    y_interp = y(:,1);
    
    % Go through each pair of channels
    for ich = 1:n_ch-1
        % Original channels
        channel1 = y(:,ich);
        channel2 = y(:,ich+1);
        % Interpolation
        n_inter = n_inters(ich);
        new_channels = channel1 + (channel2-channel1) .* [1:n_inter]/(n_inter+1);
        % Append
        y_interp = [y_interp, new_channels, channel2];
    end

end