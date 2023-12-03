clear all
clc
tic
Coe=[0.2982    0.4439    0.028];

% folder = '/Users/aditya/Downloads/Results/Images/newvid_FuNIEGAN/';
% folder = '/Users/aditya/Downloads/Results/Images/newvid_TOPAL/_300_20_1_0.1/';
% folder = '/Users/aditya/Downloads/Results/Images/newvid_UWCNN/';
% folder = '/Users/aditya/Downloads/Results/Images/newvid_UWGAN/';
% folder = '/Users/aditya/Downloads/Results/Images/newvid_WaterNet/';
% folder = '/Users/aditya/Downloads/Results/Images/newvid_PhishNet/';
% folder = '/Users/aditya/Downloads/Results/Images/newvid_PhishVideo/';
folder = '/Users/aditya/Downloads/PHiSHVideo_01_K10_UIEB/';

filepaths = dir(fullfile(folder,'*.png'));
% FDUMMetric=zeros(length(filepaths),1);
UIQMMetric=zeros(length(filepaths),1);
% UICMMetric=zeros(length(filepaths),1);
% UISMMetric=zeros(length(filepaths),1);
% UICONMMetric=zeros(length(filepaths),1);
for ii = 1 : length(filepaths)
    disp(ii)
    I=imread(fullfile(folder,filepaths(ii).name));
    Color=Colorfulness(I);
    Con=Contrast(I);
    [UIQM_norm, UICM, Sharp, UICONM] = UIQMSharpness(I);
    % Final=Coe(1)*Color+Coe(2)*Con+Coe(3)*Sharp;
    % FDUMMetric(ii,:)=Final;
    UIQMMetric(ii, :) = UIQM_norm;
    % UICMMetric(ii, :) = UICM;
    % UISMMetric(ii, :) = Sharp;
    % UICONMMetric(ii, :) = UICONM;
    % disp([filepaths(ii).name, string(UIQM_norm), string(UICM), string(Sharp), string(UICONM), string(Final)])
end
toc
% save(save_path);
mean(UIQMMetric)
% mean(UICMMetric)
% mean(UISMMetric)
% mean(UICONMMetric)
% mean(FDUMMetric)