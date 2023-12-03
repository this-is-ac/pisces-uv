clc;
clear;

output_path = "/Users/aditya/Downloads/PHiSHVideo_01_K10_UIEB/";
reference_path = "/Users/aditya/Downloads/gt/";

files = dir(fullfile(output_path,'*.png'));

names = string([]);
overall_pcqi = [];

f = waitbar(0, 'Starting');

for k = 1:length(files)
    out_file = append(output_path, files(k).name);
    ref_file = append(reference_path, files(k).name(5:end));

    names(end+1) = files(k).name(1:end-4);

    im1 = imread(ref_file);
    im2 = imread(out_file);
 
    [a,b,c] = size(im2);
    im1 = imresize(im1, 'OutputSize', [a,b]);
    
    im1=double(rgb2gray(im1));
    im2=double(rgb2gray(im2));
    
    [mpcqi,pcqi_map]=PCQI(im1,im2);

    overall_pcqi(end+1) = mpcqi;
    waitbar(k, f, sprintf('Progress: %d %%', k));
end

close(f)

disp(mean(overall_pcqi));