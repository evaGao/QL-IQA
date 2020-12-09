function writescore(scoresFile, imageName, scoreValue)

% for all patches

    % Form the image patch path string with name
    imName = imageName(1 : end - 4); % strip extension
    imPathWithName = strcat(imName, '/', imageName);
    % Write image path containing image name along with corresponding score
    % value to the output file
    fprintf(scoresFile, '%s %.4f\n', imPathWithName, scoreValue);        
end

