function ripples_keep = ripple_manual_selection(LFP, ripples, ch_ripple, fs, varargin)

    % Parse inputs 
    p = inputParser;    
	addParameter(p, 'save_name', '', @ischar);
	addParameter(p, 'shuffle', false, @islogical);
	addParameter(p, 'autosave', '', @ischar);
	addParameter(p, 'n_subplots', 10, @isnumeric); % number of rows/columns
	addParameter(p, 'win_size', 0.100, @isnumeric); % seconds
    
    parse(p,varargin{:});
    save_name = p.Results.save_name;
    shuffle = p.Results.shuffle;
    autosave = p.Results.autosave;
    n_subplots = p.Results.n_subplots;
    win_size = p.Results.win_size;
    
    % Ripples: seconds to samples
    ripples = round(ripples*fs);

    %Check autsave input validity if provided
    if ~isempty(autosave)
        [fPath, fName, fExt] = fileparts(autosave);
        %Check file extension
        if ~contains(fExt, '.mat')
            warning("Input 'autosave' does not have a '.mat' extension. Adding it.")
            fExt = '.mat';
        end
        if ~isfolder(fPath)
            warning("Indicated path in input 'autosave' does not exist. Creating it.")
            mkdir(fPath);
        end
        autosave = fullfile(fPath, [fName, fExt]);
    end

    % Make saving structure
    if ~isempty(autosave) && ~exist(autosave,'file')
        save_struct = {};
    elseif ~isempty(autosave) && exist(autosave,'file')
        load(autosave, 'save_struct')
        if isfield(save_struct, 'idxs_keep')
            idxs_keep = save_struct.idxs_keep;
        end
    end

    if ~isempty(ripples)
        
        % Number of displayed figures   
        n_figures = floor(size(ripples,1)/(n_subplots^2))+1;
        
        % Keep/Kill
        keepkillStr = {'Keep','Kill'};
        
        % Shuffle figures if shuffle=true
        if ~isempty(autosave) && exist(autosave,'file') && isfield(save_struct, 'idxs_figs')
            idxs_figs = save_struct.idxs_figs;
        else
            if shuffle
                idxs_figs = randperm(n_figures);
            else
                idxs_figs = 1:n_figures;
            end
        end

        % Save
        if ~isempty(autosave) && exist(autosave,'file') && isfield(save_struct, 'idxs_figs')
            ii = save_struct.ii;
        else
            ii = 1;
        end

        % Go through all windows
        while (ii >= 1) && (ii <= n_figures)
            ifig = idxs_figs(ii);
            
            % Figure
            f = figure('pos',[10,50,1800,800]); hold on
            
            % Supertitle
            uicontrol('style','text','string',sprintf('Window %d/%d', ii, n_figures), ...
                'unit','normalized','position',[ 0.4, 0.98 0.2 0.02 ]);
            
            % Subplots
            [ha, pos] = tight_subplot(n_subplots, n_subplots, [.0 .0],[.05 .05],[.0 .0]);
            
            % Previous figure
            uicontrol('Style', 'pushbutton', 'String', '<','Units','normalize','Position', [.030 .96 .15 .04],'Callback', @previous_figure);
            % Next figure
            uicontrol('Style', 'pushbutton', 'String', '>','Units','normalize','Position', [.830 .96 .15 .04],'Callback', @next_figure);
            
            % Keep / Kill button
            keepkill = uicontrol('Style','popupmenu', 'String', keepkillStr, 'Units','normalize', 'Position',[.78 .95 .04 .04]);
           
            % Plot ripples with checkboxs to select the ones to be deleted
            for irip = (ifig-1)*n_subplots^2+1 : min(size(ripples,1),ifig*n_subplots^2)
               isub = irip-(ifig-1)*n_subplots^2;
               axes(ha(isub)); hold on;
               cbx(isub) = uicontrol('Style','togglebutton','Units','normalize',...
                   'Position',[pos{isub}(1)+0.071,pos{isub}(2)+0.056,pos{isub}(3)*0.3,pos{isub}(4)*0.4],...
                   'CallBack', {@PushButton,f});

               % Plot
               LFP2plot = LFP( ripples(irip,1):ripples(irip,2), ch_ripple );
               plot( [ripples(irip,1):ripples(irip,2)]/fs, LFP2plot);
               
               % Axis
               xlim(mean([ripples(irip,1),ripples(irip,2)])/fs + [-win_size/2, win_size/2] )
               set(gca,'ytick',[],'xtick',[])
            end
            
            % Wait to press Done! button
            uiwait(gcf);
            
            % Keep kill selection
            keepkill_option = keepkill.Value;
            
            % Delete selected ripples
            for irip = (ifig-1)*n_subplots^2+1 : min(size(ripples,1),ifig*n_subplots^2)
               isub = irip-(ifig-1)*n_subplots^2;
               if keepkill_option==1, idxs_keep(irip) = cbx(isub).Value;
               else, idxs_keep(irip) = 1-cbx(isub).Value;
               end
            end
            
            close gcf

            % Autosave
            if ~isempty(autosave)
                save_struct.idxs_figs = idxs_figs;
                save_struct.idxRipKeep = idxs_keep;
                if ii < n_figures
                    save_struct.ii = ii;
                else
                    save_struct.ii = n_figures;
                end
                save(autosave, 'save_struct')
            end

        end
        
        
        if any(idxs_keep>0)

            % PLOT GOOD RIPPLES

            % Good ripples
            ripples_good = ripples(idxs_keep==1,:);

            % Number of subplot columns/arrays
            n_subplots = ceil(sqrt(size(ripples_good,1)));
            
            % Number of displayed figures   
            figure('pos',[10,50,1800,800]), hold on
            
            if sum(idxs_keep) < 225
                [ha, pos] = tight_subplot(n_subplots, n_subplots, [.0 .0],[.05 .05],[.0 .0]);
                % Plot 100 ripples with checkboxs to select the ones to be deleted
                for irip = 1:size(ripples_good,1)
                    isub = irip;
                    axes(ha(isub));
                    plot( [ripples_good(irip,1):ripples_good(irip,2)]/fs, 8*LFP( ripples_good(irip,1):ripples_good(irip,2), ch_ripple ) )
                    % Axis
                    xlim(mean([ripples_good(irip,1),ripples_good(irip,2)])/fs + [-win_size/2,win_size/2] )
                    set(gca,'ytick',[],'xtick',[])
                end  
            else
                % Plot all ripples
                hold on
                ysub = 0;
                for irip = 1:size(ripples_good,1)
                    if mod(irip, n_subplots) == 1
                        xsub = 0;
                        ysub = ysub - 10* mean(std(LFP(:,ch_ripple)));
                    else
                        xsub  = xsub  + 0.150;
                    end                    
                    plot( xsub + [0:(ripples_good(irip,2)-ripples_good(irip,1))]/fs, ...
                            ysub + LFP( ripples_good(irip,1):ripples_good(irip,2), ch_ripple ) )
                end  
                set(gca,'ytick',[],'xtick',[])
            end
            
            sgtitle(sprintf('%d ripples',size(ripples_good,1)))
            if ~isempty(save_name); saveas(gcf,save_name); end

            % Return
            idxs_keep = find(idxs_keep);
                
        else
            idxs_keep = [];
        end
    else
        idxs_keep = [];
    end
    
    ripples_keep = ripples(idxs_keep, :)/fs;


    % Function to go to next or previous fig
    function next_figure(~,~)
        ii = ii+1;
        uiresume(gcbf)
    end
    function previous_figure(~,~)
        ii = ii-1;
        uiresume(gcbf)
    end


end



function PushButton(hObject, EventData,F)
    if get(hObject,'value') == 1
        set(hObject,'Backgroundcolor',[1 0 0]) 
    elseif get(hObject,'value') == 0
        set(hObject,'Backgroundcolor',get(F,'color'))
    end
end