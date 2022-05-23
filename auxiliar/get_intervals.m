function intervals = get_intervals(SWR_prob, varargin)

    % Optional inputs
    p = inputParser;
    addParameter(p,'threshold_default',0.7, @isnumeric);
    addParameter(p,'threshold',nan, @isnumeric);
    addParameter(p,'LFP',[], @isnumeric);
    addParameter(p,'fs',1, @isnumeric); % Hz
    addParameter(p,'win_size',0.050, @isnumeric); % sec
    addParameter(p,'discard_drift',true, @islogical); % sec
    addParameter(p,'std_discard',1, @isnumeric); % sec
    addParameter(p, 'min_duration',0.02, @isnumeric); %sec
    parse(p,varargin{:});

    threshold_default = p.Results.threshold_default;
    threshold = p.Results.threshold;
    LFP = p.Results.LFP;
    fs = p.Results.fs;
    win_size = p.Results.win_size;
    discard_drift = p.Results.discard_drift;
    std_discard = p.Results.std_discard;
    min_duration = p.Results.min_duration;

    if ~isempty(LFP)

        % z-scored
        LFP = LFP - mean(LFP,1);
        LFP = LFP ./ std(LFP,1);

        % Discard drift
        discard = zeros(size(LFP,1), 1);

        if discard_drift

            % Moving window
            if fs == 1 
                move_win = 1000;
                warning('Sampling frequency not given, so moving window is set to 1000 samples')
            else
                move_win = win_size*fs*10;
            end
            mean_LFP_smooth = movmean(abs(mean(LFP,2)), move_win);

            % Discard times with drift above 'std_discard' standard deviations
            discard = mean_LFP_smooth > std_discard; %SD

            % Make SWR_prob zero in discarded times
            SWR_prob(discard) = 0;
        end

    end

    % === Use GUI to define threshold =====================================
    
    if isnan(threshold)

        % -- No LFP provided, examples are not plotted --------------------
        if isempty(LFP)

            % Plot
            figure('units','normalized','pos',[0.4, 0.4, 0.4, 0.3]), hold on

            % Button
            uicontrol('Style', 'pushbutton', 'String', 'Done',...
                'Units','normalize','Position', [.77 .80 .10 .10], 'Callback', 'uiresume(gcbf)');

            % Plot SWR_prob histogram
            [yhist, xhist] = hist(SWR_prob, 20);
            bar(xhist, yhist, 1, 'facecolor', [.2 .6 .8], 'edgecolor', 'none'); 
            set(gca, 'yscale', 'log', 'xlim', [0 1])
            ax = axis;
            % Plot threshold line
            threshold_plot = plot(threshold_default*[1 1], [ax(3) ax(4)], 'k', 'linewidth', 1.5);
            draggable(threshold_plot,'constraint','h');
            % Axis
            xlabel('SWR probability')
            ylabel('Distribution')

            % Wait for user to press button
            uiwait(gcf);
            
            % Get threshold value
            threshold = mean(get(threshold_plot,'XData'));
        
            
           
        % -- LFP provided, examples are plotted ---------------------------
        
        else
        
            % Plot
            fig = figure('units','normalized','pos',[0.1, 0.3, 0.8, 0.5]);
            hold on

            % Buttons
            update = uicontrol('Style', 'pushbutton', 'String', 'Update',...
                'Units','normalize','Position', [.18 .80 .05 .06], 'Callback', @update_function);
            done = uicontrol('Style', 'pushbutton', 'String', 'Done',...
                'Units','normalize','Position', [.93 .05 .05 .06], 'Callback', 'uiresume(gcbf)');

            % Threshold
            subplot(5,12,[1 2 13 14 25 26 37 38 49 50]), hold on
            % Plot SWR_prob histogram
            [yhist, xhist] = hist(SWR_prob, 20);
            bar(xhist, yhist, 1, 'facecolor', [.2 .6 .8], 'edgecolor', 'none'); 
            set(gca, 'yscale', 'log', 'xlim', [0 1])
            ax = axis;
            % Plot threshold line
            threshold_plot = plot(threshold_default*[1 1], [ax(3) ax(4)], 'k', 'linewidth', 1.5);
            draggable(threshold_plot,'constraint','h');
            % Axis
            xlabel('SWR probability')
            ylabel('Distribution')
            hold off

            % Plot examples
            update_function();
            
            % Done
            uiwait(gcf);
            
            % Get threshold value
            threshold = mean(get(threshold_plot,'XData'));

        end
            
        % Get threshold value
        threshold = mean(get(threshold_plot,'XData'));
        
    end

    % Get intervals
    intervals = get_intervals_from_threshold(SWR_prob, threshold, min_duration, fs);
    
    close;
    
    
    
    
    
    % -- Button functions -------------------------------------------------
    
    % Update function
    function update_function(~,~)
        
        % Get threshold value
        threshold = mean(get(threshold_plot,'XData'));
        
        % Get intervals
        intervals = get_intervals_from_threshold(SWR_prob, threshold, min_duration, fs);
        n_detections = size(intervals, 1);
        
        % Plot examples
        % - subplots to use
        idxs_plot = find(~ismember(1: 5*12, [1 2 13 14 25 26 37 38 49 50]));
        idxs_plot = idxs_plot(1:min(n_detections, length(idxs_plot)));
        n_plot = length(idxs_plot);
        % - random detections
        idxs_ripples_plot = sort(randperm(n_detections, n_plot));
        % - window
        if fs > 1
            idxs_win = round([-win_size/2*fs : 0.001*fs : win_size/2*fs]);
        else
            idxs_win = [-1000 : 1000];
        end
        
        % Clear subplots
        for iplot = find(~ismember(1: 5*12, [1 2 13 14 25 26 37 38 49 50]))
            sp = subplot(5,12, iplot);
            cla(sp)
            set(gca, 'xtick', [], 'ytick', [])
        end
        for iplot = 1:n_plot
            % Subplot
            sp = subplot(5,12, idxs_plot(iplot));
            % Time indexes
            imiddle = mean(intervals(idxs_ripples_plot(iplot),:)*fs);
            idxs_ripple = round(imiddle + idxs_win);
            idxs_ripple(idxs_ripple<1) = [];
            idxs_ripple(idxs_ripple>length(SWR_prob)) = [];
            plot(LFP(idxs_ripple,:)/2 - [1:size(LFP,2)], 'k')
            set(gca, 'xtick', [], 'ytick', [])
        end
        
    end
    

end



function intervals = get_intervals_from_threshold(SWR_prob, threshold, min_duration, fs)

    SWR_prob = reshape(SWR_prob, [], 1);
    % Over threshold
    over_threshold = (SWR_prob >= threshold);
    
    % Beginnings and endings
    inis = find(diff(over_threshold) > 0);
    ends = find(diff(over_threshold) < 0);
    
    % Check pairs
    if length(inis) ~= length(ends)
        if inis(1) > ends(1)
            inis = [1; inis];
        elseif inis(end) > ends(end)
            ends = [ends; length(SWR_prob)];
        end
    end
    
    % Make intervals
    intervals = [inis, ends]/fs;
    % Discard intervals shorter than min_duration
    intervals(diff(intervals,1,2)<min_duration,:) = [];
    
end








