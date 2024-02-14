clear;
clc;

filePath = "TransportationCleaned/";

ds = imageDatastore(filePath, ...
    FileExtensions=[".jpg"], ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames");
nImages = numel(ds.Files);
images = ds.Files;
labels = ds.Labels;


%% Generate training, validation, and test sets
rng("default");

% Get a list of subfolders in the main folder
subfolders = dir(filePath);
subfolders = subfolders([subfolders.isdir]); % Keep only directories
subfolders = subfolders(~ismember({subfolders.name}, {'.', '..'})); % Remove '.' and '..'

trainingPercentage = 70;
testingPercentage = 30;

trainingData = {};
testingData = {};

trainingLabels = {};
testingLabels = {};

numNumTrain = [];
numNumTest = [];

for i = 1:length(subfolders)
    subfolder = fullfile(filePath, subfolders(i).name);
    
    % Get a list of files in the current subfolder
    imgFiles = dir(fullfile(subfolder, '*.jpg'));
    
    % Randomly shuffle the list of files
    perm = randperm(length(imgFiles));

    % Calculate the number of files for each set
    numFiles = length(imgFiles);
    numTraining = round(trainingPercentage/100 * numFiles);
    numTesting = numFiles - numTraining;
    
    % Divide the files into training, validation, and testing sets
    
    for j = 1:numTraining
        trainingData = [trainingData; fullfile(subfolder, imgFiles(perm(j)).name)'];
        trainingLabels = [trainingLabels; subfolders(i).name];
    end

    for j = numTraining+1:numFiles
        testingData = [testingData; fullfile(subfolder, imgFiles(perm(j)).name)'];
        testingLabels = [testingLabels; subfolders(i).name];
    end

    disp(['Processing subfolder: ' subfolders(i).name]);
    disp(['  Training set: ' num2str(numTraining) ' files']);
    disp(['  Testing set: ' num2str(numTesting) ' files']);
    numNumTest = [numNumTest, numTesting];
    numNumTrain = [numNumTrain, numTraining];

end

%% Compute some stats
meanNumTrain = mean(numNumTrain)
meanNumTest = mean(numNumTest)

medianNumTrain = median(numNumTrain)
medianNumTest = median(numNumTest)

maxNumTrain = max(numNumTrain)
maxNumTest = max(numNumTest)

minNumTrain = min(numNumTrain)
minNumTest = min(numNumTest)

%% Generate datastores
imgTrain = imageDatastore(trainingData);

imgTest = imageDatastore(testingData);

%% Load net and augment inputs 

net = resnet101;

inputSize = net.Layers(1).InputSize;

%% Augment Inputs

%you could experiment here with different augmentations maybe?
% imageAugmenter = imageDataAugmenter('RandXTranslation', [-20 20], 'RandYTranslation', [-20 20], 'RandRotation', [-180 180], 'RandScale', [0.5 2]);
% augimdsTrain = augmentedImageDatastore(inputSize(1:2), imgTrain, 'DataAugmentation', imageAugmenter);
augimdsTrain = augmentedImageDatastore(inputSize(1:2), imgTrain);
augimdsTest = augmentedImageDatastore(inputSize(1:2), imgTest);

%% Feature Extraction with early activation layer: res3b3_relu features

layer = 'res3b3_relu';
featuresTrain = squeeze(mean(activations(net,augimdsTrain,layer),[1 2]))';
featuresTest = squeeze(mean(activations(net,augimdsTest,layer),[1 2]))';

%% Train multi-class SVM

svmTemplate = templateSVM('BoxConstraint',0.0067609,'KernelFunction', 'rbf','KernelScale',0.17227,'Standardize', false);
classifier = fitcecoc(featuresTrain,trainingLabels, 'Learners',svmTemplate);

%% Checkpoint save
save('res3b3featuresClassifier.mat');

%% Checkpoint load
load('res3b3featuresClassifier.mat')

%% Predictions and etc.

[YPred, score] = predict(classifier,featuresTest);
accuracy = mean(YPred == testingLabels)

temp1 = table(YPred, testingLabels, max(score), augimdsTest.Files);
temp1

