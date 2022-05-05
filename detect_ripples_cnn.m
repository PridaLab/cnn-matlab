function y_pred = detect_ripples_cnn(data, fs, varargin)

%%%  Uses the CNN model to predict ripples of a given data
%%%
%%% Inputs:
%%%    data         Time x channels LFP matrix
%%%    fs           Sampling frequency
%%%    exec_env     (optional) Full path to python executable of the
%%%                 environment containing tensorflow. By default is
%%%                     /home/andrea/anaconda3/envs/rip_env/bin/python3.7
%%%    model_file   (optional) Full path to folder where the CNN model is
%%%                 stored. By default searches in 'cnn/' inside the folder
%%%                 containing this file.
%%%    channels     (optional) List of 8 channels to use for the
%%%                 prediction. By default it takes the first 8 channels
%%%    threshold    (optional) 0.5 by default
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
    addParameter(p,'threshold', 0.5, @isnumeric);
    addParameter(p,'model_file', '', @ischar);
    addParameter(p,'exec_env', '/home/andrea/anaconda3/envs/tfenv37/bin/python3.7', @ischar);
    addParameter(p,'save', {}, @isstruct);
    parse(p,varargin{:});
    channels = p.Results.channels;
    threshold = p.Results.threshold;
    model_file = p.Results.model_file;
    exec_env = p.Results.exec_env;
    dir_project = fileparts(matlab.desktop.editor.getActiveFilename);
    
    % Model file
    if isempty(model_file)
        model_file = fullfile(fileparts(matlab.desktop.editor.getActiveFilename), 'cnn');
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
    y_pred = single(py.cnn.func(data, channels, fs, model_file, threshold));


end

