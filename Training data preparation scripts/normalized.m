function normalizedImage = normalized(imageresize, windowSize)
% pre-allocate output memory

% Define local neighbourhood
nHood = ones(windowSize, windowSize);
% Define local mean filter
meanFilter = ones(windowSize, windowSize) / (windowSize * windowSize);

% Run through all patches
patch = imageresize;
        % Compute local mean
meanPatch = conv2(patch, meanFilter, 'same');
        % Compute local standard deviation
stdDevPatch = stdfilt(patch, nHood);
        % Subtract local mean and divide by local standard deviation and add a
        % small constant to denominator in order to avoid division by zero
normalizedImage = (patch - meanPatch) ./ (stdDevPatch + 1e-8);
  

end
