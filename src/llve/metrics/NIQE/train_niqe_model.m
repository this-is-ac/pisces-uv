imds = imageDatastore(fullfile('gt'),'FileExtensions',{'.png'});

model = fitniqe(imds);