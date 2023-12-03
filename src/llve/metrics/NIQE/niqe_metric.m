% folderpath = "Results/Images/newvid_FuNIEGAN/";
% folderpath = "Results/Images/newvid_TOPAL/_300_20_1_0.1/";
% folderpath = "Results/Images/newvid_UWCNN/";
% folderpath = "Results/Images/newvid_UWGAN/";
% folderpath = "Results/Images/newvid_WaterNet/";
folderpath = "Results/Images/newvid_PhishNet/";
% folderpath = "Results/Images/newvid_PhishVideo/";

imagefiles = dir(folderpath + "*.png");
all_niqe = [];
nfiles = length(imagefiles);

load('custNIQE', 'model');

for ii=1:nfiles
   disp(ii);
   currentfilename = imagefiles(ii).name;
   currentimage = imread(folderpath + currentfilename);
   all_niqe(end+1) = niqe(currentimage, model);
end

disp(mean(all_niqe));
