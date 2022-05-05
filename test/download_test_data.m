function [lfp, fs, ripples] = download_test_data()
    
    test_path = fileparts(matlab.desktop.editor.getActiveFilename);

    % --- Download example data -----------
    data_path = fullfile(test_path, 'data');
    if ~exist(data_path, 'dir')
        disp('Downloading LFP from Prida figshare, this may take a while...')
        % Make "data" folder
        mkdir(data_path)
        % Download data from figshare
        options = weboptions('Timeout',Inf);
        websave(fullfile(data_path,'lfp.dat'),'https://figshare.com/ndownloader/files/28813914') % LFP
        websave(fullfile(data_path,'info.mat'),'https://figshare.com/ndownloader/files/28813911') % fs
        websave(fullfile(data_path,'ripples.csv'),'https://figshare.com/ndownloader/files/28813962') % SWR
    end


    % Read fs
    load(fullfile(data_path,'info.mat'), 'fs')

    % Read LFP
    lfp = double(bz_LoadBinary(fullfile(data_path, 'lfp.dat'),'nChannels',8));

    % Read ground truth
    ripples = readtable(fullfile(data_path, 'ripples.csv'));
    ripples = [ripples.ripIni ripples.ripEnd]/fs;

end