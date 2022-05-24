function y_interp = interpolate_channels(y, ch_map)

%%%  Adds interpolated channels between any pair
%%%
%%%  >> y_interp = interpolate_channels(y, ch_map)
%%%
%%% Inputs:
%%%    y            n_samples x n_channels LFP matrix

%%%    ch_map       vector of size (1 x n_desired_channels). The channels used
%%%                 as a source are indicated by the number of the channel (>0),
%%%                 and the desired interpolated ones by a -1.
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
%%%          >> y_interp = interpolate_channels(y, [1 -1 -1 2 -1 3 -1 4])
%%%
%%%     Let's say now that size(y) = [N, 8], but channel 6 is dead and we want
%%%     to interpolte it. Then we could do:
%%%          >> y_interp = interpolate_channels(y, [1 2 3 4 5 -1 7 8])
%%% 
%%% A. Navas-Olive & Julio E, LCN 2022

    if all(ch_map>-1)
        warning('All channels provided, no interpolation done. Just rearranging the channels.')
    end
    if size(ch_map,1)>size(ch_map,2)
        ch_map = ch_map';
    end
    %Check that dimensions fit
    if size(ch_map,1)>1
        error("Input 'ch_map' must be a 1D vector, but it was (%i,%i).", ...
            size(ch_map,1),size(ch_map,2))
    end
    %Check that first and last channels are provided
    if ch_map(1,1)<1 || ch_map(1,end)<1
        error("First and last channels of input 'ch_map' must be provided (>0).")
    end
    %Check that provided channels exist in y
    if any(ch_map>size(y,2)) || any(ch_map==0)
        error("Provided channels in 'ch_map' are not valid. They must be between [1, size(y,2)].")
    end

    y_interp = zeros(size(y,1), size(ch_map,2));
    for ch_idx = 1:size(ch_map,2)
        ch_val = ch_map(1,ch_idx);
        if ch_map(1,ch_idx)>-1
            y_interp(:, ch_idx) = y(:,ch_val);
        else
            %find previous provided channel
            pre_ch_idx = find(ch_map(1,1:ch_idx-1)>-1);
            pre_ch_idx = pre_ch_idx(end);
            pre_ch_val = ch_map(1,pre_ch_idx);

            %find incoming provided channel
            post_ch_idx = find(ch_map(1,ch_idx+1:end)>-1);
            post_ch_idx = post_ch_idx(1)+ch_idx;
            post_ch_val = ch_map(1,post_ch_idx);

            %do weighted mean
            ch_dist = post_ch_idx - pre_ch_idx; 
            y_interp(:, ch_idx) = y(:, pre_ch_val) + ((ch_idx - pre_ch_idx)/(ch_dist))*...
                                                    (y(:, post_ch_val) - y(:, pre_ch_val));
        end
    end

end