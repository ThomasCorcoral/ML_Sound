# Wildlife Identification Learning Tools
#### Audio Recording Classification by Machine Learning

#### Link to download UrbanSound8K to test the program : https://goo.gl/8hY5ER

#### Link to download bird dataset (under construction) : https://mega.nz/file/ong2nBYY#FMvn2lxBZrS7bBD0sABSQ9z6Q5GOcYF0agXG_Y4t52Y

## Train

To use .mp3 please: <br>
WINDOWS :
<ul>
  <li>Fill in the path to the absolute path /ffmpeg folder in the PATH environment variable </li>
  <li>Unzip the ffmpeg.7z file in the ffmpeg</li>
</ul>
<br>
  LINUX :
<ul>
  <li>Install ffmpeg throught `sudo apt install ffmpeg` </li>
</ul>
<br>
`cnn.py` This file contains the model and will automatically perform the training with the paths entered in the `convert_data.py` file<br>
<br>
`app.py` GUI application that makes it easy and easy to use deeplearning to detect an unknown record from a database<br>

## Test

`prediction.py` This file allows you to make a prediction using a pre-trained model

`app.py` The GUI app allows you once your data is prepared and your model is trained to classify audio recording

## GUI application

In the directory `./application` you'll find `app.py` which will allow you to launch the application that will allow you to prepare your data and train the model. It is also possible to view (audio, spectrogram and mfcc) the file you want to classify.

## Misc

### Organisation

In order to organize our project we first created this github as well as a <a target="_blank" href="https://trello.com/b/n5JMlDKU/conduite-de-projet" title="Site">Trello</a>. A website followed in order to be able to access easily and at any time a lot of information (see below).

### Usage

To be able to use the application, install python 3.8 and the dependencies present in the file requirements.txt. Once this is done, launch the app.py file, for more information click on the "Help" button at the bottom right of the application

### Application

The application developed in python uses the tkinter library to perform the following display:

![Preview](https://github.com/ThomasCorcoral/Projet_L3/blob/main/img/screen_application.png)


### Website

#### <a target="_blank" href="https://projet.xnh.fr/index.html" title="Site">Visit it</a>

  This website contains :
<ul>
  <li>A training section to learn the basics to contribute to the project</li>
  <li>Different resources + meeting minutes</li>
  <li>Reminder of links to <a target="_blank" href="https://trello.com/b/n5JMlDKU/conduite-de-projet" title="Site">Trello</a>, Github and <a target="_blank" href="https://docs.google.com/document/d/1nI-bLGr7N6MVG3OC4NAJBIDIE_Ea6VqnJSBD6i3UwQ4/edit" title="Site">Google Doc</a></li>
</ul>
  
### Dependencies 
##### `requirements.txt`

`librosa~=0.8.0`<br>
`matplotlib~=3.3.3`<br>
`numpy~=1.19.4`<br>
`scipy~=1.5.4`<br>
`tensorflow~=2.4.0`<br>
`scikit-learn~=0.23.2`<br>
`Keras~=2.4.3`<br>
`pandas~=1.1.5`<br>

### References

```
@misc{smales_2020, title={Sound Classification using Deep Learning}, url={https://mikesmales.medium.com/sound-classification-using-deep-learning-8bc2aa1990b7}, journal={Medium}, publisher={Medium}, author={Smales, Mike}, year={2020}, month={Nov}}
```
```
@inproceedings{batdetect18, title={Bat Detective - Deep Learning Tools for Bat Acoustic Signal Detection}, author={Mac Aodha, Oisin and Gibb, Rory and Barlow, Kate and Browning, Ella and Firman, Michael and   Freeman, Robin and Harder, Briana and Kinsey, Libby and Mead, Gary and Newson, Stuart and Pandourski, Ivan and Parsons, Stuart and Russ, Jon and Szodoray-Paradi, Abigel and Szodoray-Paradi, Farkas and Tilova, Elena and Girolami, Mark and Brostow, Gabriel and E. Jones, Kate.}, journal={PLOS Computational Biology}, year={2018}}
```
```
@misc{gavali_mhetre_patil_bamane_buva_2019, title={Bird Species Identification using Deep Learning}, url={https://www.ijert.org/bird-species-identification-using-deep-learning#:~:text=200 [CUB-200-2011]) shows that algorithm achieves,between 80% and 90%.&text=Birds help us to detect,the environmental changes [2].}, journal={International Journal of Engineering Research & Technology}, publisher={IJERT-International Journal of Engineering Research & Technology}, author={Gavali, Prof. Pralhad and Mhetre, Ms. Prachi Abhijeet and Patil, Ms. Neha Chandrakhant and Bamane, Ms. Nikita Suresh and Buva, Ms. Harshal Dipak}, year={2019}, month={Apr}}
```
```
@misc{kortas_2020, title={Sound-Based Bird Classification}, url={https://towardsdatascience.com/sound-based-bird-classification-965d0ecacb2b}, journal={Medium}, publisher={Towards Data Science}, author={Kortas, Magdalena}, year={2020}, month={Jan}}
```
```
@misc{salamon_jacoby_bello_new york_new york, title={UrbanSound8K}, url={https://urbansounddataset.weebly.com/urbansound8k.html}, journal={Urban Sound Datasets}, author={Salamon, Justin and Jacoby, Christopher and Bello, Juan Pablo and New York, MARL and New York, CUSP}}
```
```
@misc{bushaev_2018, title={Adam - latest trends in deep learning optimization.}, url={https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c#:~:text=Adam [1] is an adaptive,for training deep neural networks.&text=The algorithms leverages the,learning rates for each parameter}, journal={Medium}, publisher={Towards Data Science}, author={Bushaev, Vitaly}, year={2018}, month={Oct}}
```
