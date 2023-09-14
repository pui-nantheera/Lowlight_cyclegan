% FOR TESTING

clear all

% parameters
maxcount = 2000;
patchsize = 512;
expand = floor((1500 - patchsize)/2);
type = 'testA'; 
dataFolder = 'W:\ESPRIT\Helen Hobin BBC_frames\NR Test Shots\Moving Subject Moving Camera\ND1.2\';
dataName = 'H001_C011_07126H_001_ISO100';

outBase = 'W:\datasets\patches\';

outFolder = [outBase,dataName,'\', type,'\'];
outFolderreg = [outBase,dataName,'\', type,'_reg\'];
mkdir(outFolder);
mkdir(outFolderreg);

files = dir([dataFolder, dataName,'\*.tif']);

for k = 1:length(files)
 
    img = im2double(imread(fullfile(files(k).folder, files(k).name)));
    filename = ['f',  sprintf('%05d',k)];
    
    % start position
    for x = 1:patchsize/2:size(img,2)%-patchsize+1
        for y = 1:patchsize/2:size(img,1)%-patchsize+1
            outName = [filename,'_x',num2str(x),'_y',num2str(y),'.png'];
            xind = x;
            yind = y;
            xend = xind+patchsize-1;
            yend = yind+patchsize-1;
            reg = img(max(1,yind-expand):min(size(img,1),yend+expand), max(1,xind-expand):min(size(img,2),xend+expand),:);
            imwrite(imresize(reg, [patchsize, patchsize]), [outFolderreg, outName]);
            
            reg = img(max(1,yind):min(size(img,1),yend), max(1,xind):min(size(img,2),xend),:);
            imwrite(reg, [outFolder, outName]);
        end
    end
end
