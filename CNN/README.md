# Structure of CNN:

### A Convolutional Neural Network for Image Classification contains following images:

1. CNN Layers: This layer extracts features and patterns from the imput images.
2. Pooling Layer: This layer reduces the dimensionality by summarizing the neighbouring pixels.
3. Flatten Layer: This layer converts the matrics of features obtained into 1-D array.
4. Classification Layer: This layer takes a 1-D array as an input from Flatten layer and performs the image classification.

**Convolutional Layer: A Convolutional Neural Network with some convolutional layers(and some other layers). A Convolutional layer has a number of filters that does convolutional operation.**

When a (6 by 6) matrix convolved with a (3 by 3) matrix(filter/kernel) then the result will be a (4 by 4) matrix.
Filters will in a specific based on the work we are doing for find the Horizontal edges we need to use one kind of filters and for evaluating the vertical edges we need to use different kind of filters. Filters are also called as kernels.

The early layers of Neural Networks might detect edges and then the some later layers might detect the cause of objects and then even the later layers may detects cause of complete objects like people's faces.

We can even define the filter size. It basically depends on what kind of features we are trying to extract from the image.

**During a convolution operation there involves a image matrix, a filter and a stride.**

![Sample Image](images/1.jpeg)

If we keep on increasing the number of filters then the dimensionality of the data will get increased. Count of the filters has to be reduced so that complexity of the model doesn't increases and the feature extraction process is a bit faster.

### Activation Functions:

![Sample Image](images/2.jpeg)




