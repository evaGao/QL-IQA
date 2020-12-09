function save(NomalizedImage, outputDir, imageName)

patchesOutputDir = strcat(outputDir, imageName(1 : end - 4), '/');
% if the directory does not exist then create it
if(exist(patchesOutputDir, 'dir') == 0)
    mkdir(patchesOutputDir);
end

% Run through all patches 
outFile = strcat(patchesOutputDir, imageName);
        % Save to disk
imwrite(NomalizedImage, outFile);
   
end
