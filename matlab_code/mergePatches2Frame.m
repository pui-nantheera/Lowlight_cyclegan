clear all

hpatch = 512;%
wpatch = hpatch;
overlapRatio = 0.5;
hgap = round(hpatch*overlapRatio);
wgap = round(wpatch*overlapRatio);
a = normpdf(1:hpatch,hpatch/2,hpatch/6);
b = normpdf(1:wpatch,wpatch/2,wpatch/6);
wmap = a'*b;
wmap = wmap/max(wmap(:));
wmap = repmat(wmap,[1 1 3]);

height = 2700;
width = 5120;

epoch = '200';
destName=['/results/test_',epoch,'/images/*fake_B.png'];
resultFolder = ['/results/test_',epoch,'/frames/'];
mkdir(resultFolder)

% merge pathes to frames
files = dir([destName, '*.png']);
for k = 1:length(files)
    if ~isempty(strfind(files(k).name,'_x'))
        indx = strfind(files(k).name,'_x');
        indy = strfind(files(k).name,'_y');
        ind = strfind(files(k).name(indy+1:end),'.png');
        indye = indy+ind(1)-1;
        
        if k == 1
            frame = zeros(height, width, 3);
            weightmap = zeros(height, width, 3);
        else
            if ~strcmp(previousname,files(k).name(1:indx-1))
                frame = frame./(weightmap+10^-10);
                imwrite((frame), [resultFolder, epoch,'_', previousname, '.png'])
                %img = imresize(frame,400/size(frame,1));
                %imwrite((img), [resultFolder, epoch,'_', previousname, '.jpg'],'Quality',100)
                frame = zeros(height, width, 3);
                weightmap = zeros(height, width, 3);
            end
        end
        yind = str2double(files(k).name(indx+2:indy-1));
        xind = str2double(files(k).name(indy+2:indye));
        xind = round(xind);
        yind = round(yind);
        
        
        rangei = yind:min(height,yind+hpatch-1);
        rangej = xind:min(width,xind+wpatch-1);
        ri = 1:length(rangei);
        rj = 1:length(rangej);
        
        patch = im2double(imread(fullfile(files(k).folder, files(k).name)));
        [patchsizey, patchsizex, patchsized] = size(patch);
        if (patchsizey~=length(rangei)) || (patchsizex~=length(rangej))
            patch = imresize(patch, [length(rangei) length(rangej)]);
        end
        
        frame(rangei,rangej ,:)  = frame(rangei, rangej,:) + patch.*wmap(ri,rj,:);
        weightmap(rangei,rangej,:) = weightmap(rangei,rangej,:) + wmap(ri,rj,:);
        
        previousname = files(k).name(1:indx-1);
    end
end
frame = frame./(weightmap+10^-10);
imwrite((frame), [resultFolder, epoch,'_', previousname, '.png'])
%img = imresize(frame, 400/size(frame,1));
%imwrite((img), [resultFolder, epoch,'_', previousname, '.jpg'],'Quality',100);

