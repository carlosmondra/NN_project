# Using FCNs to identify roads and cars from satellite imagery to estimate traffic level

We aim to semantically segment a road and cars in an image by implementing a Fully Convolutional Network that is, given an image, we are able to predict which pixels of the image are roads and cars. We use 125 training images of satellite imagery from the Google Maps API and create the training masks manually. We also use transfer learning and VGGNet trained on a classification dataset to boost accuracy. Finally, we propose a simple way to identify the level of traffic in a certain image.

## Getting Started

Use
```
$ git clone https://github.com/clankster99/NN_project.git
```
to clone the project to your local machine.

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

### And coding style tests

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

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Carlos Daniel Mondragon Chapa**
* **Salar Satti**

See list of [contributors](https://github.com/clankster99/NN_project/graphs/contributors) who participated in this project.

## License

This project is licensed under the MIT License.

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
