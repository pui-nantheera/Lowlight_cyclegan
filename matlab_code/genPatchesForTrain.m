
clear all

% parameters
maxcount = 2000;
patchsize = 360;
expand = floor((1500 - patchsize)/2);
type = 'trainA'; %'trainA' if dark, 'trainB' if bright
dataFolder = 'W:\ESPRIT\Helen Hobin BBC_frames\NR Test Shots\Moving Subject Moving Camera\ND1.2\';
dataName = 'H001_C011_07126H_001_ISO100';

%type = 'trainB';
%dataFolder = 'W:\ESPRIT\Helen Hobin BBC_frames\NR Test Shots\Moving Subject Moving Camera\NOND\';
%dataName = 'H001_C011_07126H_001_ISO800';

outBase = 'W:\datasets\patches\';

outFolder = [outBase,dataName,'\', type,'\'];
outFolderreg = [outBase,dataName,'\', type,'_reg\'];
mkdir(outFolder);
mkdir(outFolderreg);

files = dir([dataFolder, dataName,'\*.tif']);
img = im2double(imread(fullfile(files(1).folder, files(1).name)));
[height, width, depth] = size(img);

% get ind of random frames and positions
framerand = [];
for k = 1:ceil(maxcount/length(files))
    framerand = [framerand (1:length(files))];
end
framerand = framerand(randperm(maxcount));
xrand = randperm(width-patchsize, maxcount);
yrand = randperm(height-patchsize, maxcount);

% sort frames
[framerand, ind] = sort(framerand);
xrand = xrand(ind);
yrand = yrand(ind);

prevk = 0;
for count = 1:maxcount
    
    k = framerand(count);
    y = yrand(count);
    x = xrand(count);
    
    if prevk ~= k
        disp(num2str(k));
        % read image
        img = im2double(imread(fullfile(files(k).folder, files(k).name)));
    end
    
    outName = [files(k).name(1:end-4),'_x',num2str(x),'_y',num2str(y),'.png'];
    imwrite(img(y:y+patchsize-1, x:x+patchsize-1,:), [outFolder, outName]);
    
    xind = x;
    yind = y;
    xend = xind+patchsize-1;
    yend = yind+patchsize-1;
    reg = img(max(1,yind-expand):min(height,yend+expand), max(1,xind-expand):min(width,xend+expand),:);
    imwrite(imresize(reg, [patchsize, patchsize]), [outFolderreg, outName]);
    
end


disp('Done!!');
