%filePath = "E:\Logos\LogoDet-3K\Transportation";
filePath = "C:\Users\kingnm\Documents\MATLAB\Image Recognition\Logo-Recognition\RegionRecognition\Transportation\";
imds = imageDatastore(filePath, "FileExtensions", [".jpg"], "IncludeSubfolders", true, "LabelSource", "foldernames");
classes = imds.Labels;
classes = unique(classes, "rows", "stable");

numFiles = numel(imds.Files);
% Generate random permutation indices for shuffling
perm = randperm(numel(imds.Files));

% Shuffle the indices and use them to access the files
shuffledFiles = imds.Files(perm);
shuffledLabels = cellstr(imds.Labels(perm));

% Calculate the number of files for each split
numTrain = round(0.6 * numFiles);
numValidation = round(0.1 * numFiles);

bbds = fileDatastore(filePath, "ReadFcn", @readXML, "IncludeSubfolders", true, "FileExtensions", [".xml"]);

bboxes = readall(bbds);
shuffledValues = bboxes(perm);
Labels = shuffledLabels(1:numTrain);
filename = shuffledFiles(1:numTrain);
Boxes = shuffledValues(1:numTrain);

%trainTable = table(filename, logo);
trainTable = table(Boxes, Labels);

imds = imageDatastore(filename);
blds = boxLabelDatastore(trainTable);
%lbls = arrayDatastore(label);

trainds = combine(imds,blds);

inputSize = [224 224 3];
trainingDataForEstimation = transform(trainds,@(data)preprocessData(data,inputSize));

numAnchors = 6;
[anchors, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation,numAnchors);
area = anchors(:,1).*anchors(:,2);
[~,idx] = sort(area,"descend");
anchors = anchors(idx,:);
anchorBoxes = {anchors(1:3,:);anchors(4:6,:)};

detector = yolov4ObjectDetector("tiny-yolov4-coco",classes,anchorBoxes,InputSize=inputSize);

options = trainingOptions("sgdm", ...
    InitialLearnRate=0.001, ...
    MiniBatchSize=32,...
    MaxEpochs=5, ...
    Plots="training-progress", ...
    BatchNormalizationStatistics="moving",...
    ResetInputNormalization=false,...
    VerboseFrequency=30);

trainedDetector = trainYOLOv4ObjectDetector(trainds,detector,options);

I = imread(shuffledFiles{numTrain + 4, 1});

[bboxes, scores, labels] = detect(trainedDetector,I,Threshold=0.05);
detectedImg = insertObjectAnnotation(I,"Rectangle",bboxes,labels);
figure
imshow(detectedImg)

function data = readXML(filename) 

    xml = readstruct(filename).object.bndbox;
    xmin = xml.xmin;
    ymin = xml.ymin;
    width = xml.xmax - xml.xmin;
    height = xml.ymax - xml.ymin;

    if xmin == 0
        xmin = 1;
    end
    if ymin == 0
        ymin = 1;
    end

    data = [xmin ymin width height];

end

function data = preprocessData(data,targetSize)
for num = 1:size(data,1)
    I = data{num,1};
    imgSize = size(I);
    bboxes = data{num,2};
    I = im2single(imresize(I,targetSize(1:2)));
    scale = targetSize(1:2)./imgSize(1:2);
    bboxes = bboxresize(bboxes,scale);
    data(num,1:2) = {I,bboxes};
end
end