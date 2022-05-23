function y_pred = detect_ripples_cnn(data, fs, varargin)

%%%  Uses the CNN model to predict ripples of a given data
%%%
%%% Inputs:
%%%    data             Time x channels LFP matrix
%%%    fs               Sampling frequency
%%%    exec_env         (optional) Full path to python executable of the
%%%                     environment containing tensorflow. By default is
%%%                         /home/andrea/anaconda3/envs/rip_env/bin/python3.7
%%%    model_file       (optional) Full path to folder where the CNN model is
%%%                     stored. By default searches in 'cnn/' inside the folder
%%%                     containing this file.
%%%    channels         (optional) List of 8 channels to use for the
%%%                     prediction. By default it takes the first 8 channels
%%%    pred_every       (optional)  Time window of predictions. By default is
%%%                     32ms, for which CNN works significantly fastest. If 
%%%                     smaller sliding windows are preferred, then any other 
%%%                     number can be selected.
%%%    handle_overlap   (optional) In order to handle prediction of overlapping 
%%%                     windows, choose to do the 'mean' (by default) or 'max'
%%%                     By default is false
%%%    verbose          (optional) Print description of internal processes. 
%%%                     By default is false
%%%
%%% Output:
%%%    y_pred       Ripple probability (from 0 to 1) per sample
%%% 
%%% R. Amaducci and A. Navas-Olive, LCN 2022

    % -----------------------
    %   Get variables
    % -----------------------    
    
    % Get optional values
    p = inputParser;
    addParameter(p,'channels', 1:8, @isnumeric);
    addParameter(p,'model_file', '', @ischar);
    addParameter(p,'pred_every', 0.032, @isnumeric);
    addParameter(p,'verbose', false, @islogical);
    addParameter(p,'handle_overlap', 'mean', @ischar);
    addParameter(p,'exec_env', '/home/andrea/anaconda3/envs/tfenv37/bin/python3.7', @ischar);
    parse(p,varargin{:});
    channels = p.Results.channels;
    pred_every = p.Results.pred_every;
    verbose = p.Results.verbose;
    model_file = p.Results.model_file;
    handle_overlap = p.Results.handle_overlap;
    exec_env = p.Results.exec_env;
    dir_project = fileparts(which('detect_ripples_cnn.m'));
    

    % Model file
    if isempty(model_file)
        model_file = fullfile(dir_project, 'cnn');
    end
    
    % -----------------------
    %   Add paths
    % ----------------------- 

    % Import python
    [~, ~, isloaded] = pyversion;
    %[~, ~, isloaded] = pyenv;
    if ~isloaded
        version = ver('MATLAB'); 
        version.Release;
        % If matlab version is > 2019, use pyenv
        if str2double(version.Release(3:6)) >= 2019
            pe = pyenv('Version', exec_env);
        % For earlier versions, use pyversion and check if is already loaded
        else
            eval(sprintf('pyversion %s',exec_env))
        end
    end
    %py.sys.setdlopenflags(int32(10)); % Fix incompatibilty between hdf5 versions https://www.mathworks.com/matlabcentral/answers/345709-how-to-call-python-h5py-from-windows-matlab-works-in-os-x-and-linux

    % Import proyect paths
    insert(py.sys.path, int32(0), dir_project);
    
    % Import script
    mod = py.importlib.import_module('cnn'); 
    py.importlib.reload(mod); 
    
    % -----------------------
    %   Predict
    % ----------------------- 

    % Predict
    y_pred = double(py.array.array('f',py.cnn.predict(data, channels, fs, model_file, pred_every, handle_overlap, verbose)));


end

