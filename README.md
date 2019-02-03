# Using FCNs to identify roads and cars from satellite imagery to estimate traffic level

We aim to semantically segment a road and cars in an image by implementing a Fully Convolutional Network that is, given an image, we are able to predict which pixels of the image are roads and cars. We use 125 training images of satellite imagery from the Google Maps API and create the training masks manually. We also use transfer learning and VGGNet trained on a classification dataset to boost accuracy. Finally, we propose a simple way to identify the level of traffic in a certain image.

## Getting Started

Use
```
$ git clone https://github.com/clankster99/NN_project.git
```
to clone the project to your local machine.

We know explain what the main python files do in this project.

roads_cars.py uses transfer learning from VGG16 to train whether to identify roads or cars. roads_cars.py is run using the following arguments:

'type': choices=['roads','cars'], help="Choose the type of model to train/load - either a model for cars or roads"
'-v', help="When passing v, a model will be loaded and validated using validation images. If this argument is not passed, then a new model will be trained and stored in the models folder."
'-p', help="Persist image."
'-s', help="Show image in a pop-up window."

no_pretrain_roads_cars.py uses an FCN architecure to train whether to identify only roads and trains it from scratch. This does not train a model to identify cars since the other approach much works better. roads_cars.py is run using the following arguments:

'architecture', choices=['1','2'], help="Choose the architecture to use 1 is nearest neighbour upsampling and 2 is transposed convolution for the upsamplling procedure"
'-v', help="When passing v, a model will be loaded and validated using validation images. If this argument is not passed, then a new model will be trained and stored in the models folder."
'-p', help="Persist image."
'-s', help="Show image."

predict_traffic.py takes the pixels where they are predicted roads and the pixels where they are predicted cars and determines what percentage of the road is filled with cars to get the level of traffic.

### Prerequisites

You need to have Python and Pytorch installed. We recommend using Anaconda to install the project.

### Installing

The installation text file contains all the dependencies needed to run the project. You can use conda to install all of them from the file requirements.txt by using the following command.
(Missing the installation text file).
```
$ conda install --yes --file requirements.txt
```

## Running the code

The project contains some pretrained models that can be run

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.


## Authors

* **Carlos Daniel Mondragon Chapa**
* **Salar Satti**

See list of [contributors](https://github.com/clankster99/NN_project/graphs/contributors) who participated in this project.

## License

This project is licensed under the MIT License.

