inputDir = "E:\Logos\LogoDet-3K\TransportationCleaned";
outputDir = "E:\Logos\CroppedCleaned";

imds = imageDatastore(inputDir, "FileExtensions", [".jpg"], "IncludeSubfolders", true, "LabelSource", "foldernames");
bbds = fileDatastore(inputDir, "ReadFcn", @readXML, "IncludeSubfolders", true, "FileExtensions", [".xml"]);
bboxes = readall(bbds);
imageFiles = imds.Files; 

% Create output directory if it does not exist
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Process each image and save the cropped version
for i = 1:numel(imageFiles)
    % Read the original image
    imagePath = imageFiles{i, 1};
    originalImage = imread(imagePath);

    % Get the bounding box for the current image
    rect = bboxes{i};
    padding = 20;
    box = [rect(1) - padding; rect(2) - padding; rect(3) + padding * 2; rect(4) + padding * 2];

    % Crop the image based on the bounding box
    croppedImage = imcrop(originalImage, box);

    % Create the subfolder structure in the output directory
    [path, name, ext] = fileparts(imageFiles{i, 1});
    [~, parent, ~] = fileparts(path);

    newPath = fullfile(outputDir, parent);

     if ~exist(newPath, 'dir')
        mkdir(newPath);
    end

    outputImagePath = fullfile(outputDir, parent, [name, '_cropped.png']);
    imwrite(croppedImage, outputImagePath);
end

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

