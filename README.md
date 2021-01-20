# Outils d’apprentissage pour l’identification de faune sauvage
#### Classification d'enregistrement audio par deeplearning

#### Lien pour télécharger UrbanSound8K pour tester le programme : https://goo.gl/8hY5ER

## Train

`cnn.py` Ce fichier contient le modèle et va automatiquement réaliser l'entrainement avec les path renseigné dans le fichier `convert_data.py`

`app.py` Application GUI qui permet d'utiliser simplement et facilement le deeplearning afin de détecter un enregistrement inconnu à partir d'une base de données

## Test

`prediction.py` Ce fichier permet de réaliser une prédiction à l'aide d'un modèle pré-entrainé

`app.py` L'application GUI vous permet une fois votre data préparé et votre modèle entrainé de classifier l'enregistrement audio

## GUI application

Dans le dossier `./application` vous trouverez le fichier `app.py` qui vous permettra de lancer l'application qui vous permettra de pouvoir préparer vos données et d'entrainer le modèle. Il est également possible de visualiser (audio, spectrogramme et mfcc) le fichier que vous voulez classifier.

## Misc

### Organisation

Afin d'organiser notre projet nous avons tout d'abord créé ce github ainsi qu'un <a target="_blank" href="https://trello.com/b/n5JMlDKU/conduite-de-projet" title="Site">Trello</a>. S'en est suivi un site internet afin de pouvoir accéder facilement et à tout moment à bon nombre d'informations (voir ci-dessous).

### Utilisation

Afin de pouvoir utiliser l'application, installez python 3.8 et les dépendances présentes dans le fichier requirements.txt. Une fois cela fait, lancez le fichier app.py, pour plus d'informations cliquez sur le bouton "Aide" en bas à droite de l'application

### Application

L'application développé en python utilise la bibliothèque tkinter afin de réaliser l'affichage suivant :

![Preview](https://github.com/ThomasCorcoral/Projet_L3/blob/main/img/screen_application.png)


### Site

#### <a target="_blank" href="https://projet.xnh.fr/index.html" title="Site">Accéder au site</a>

  Ce site internet contient :
<ul>
  <li>Une section Training permettant d'apprendre les bases pour contribuer au projet</li>
  <li>Différentes ressources + comptes rendues de réunion</li>
  <li>Le rappel des liens vers le <a target="_blank" href="https://trello.com/b/n5JMlDKU/conduite-de-projet" title="Site">Trello</a>, le Github et le <a target="_blank" href="https://docs.google.com/document/d/1nI-bLGr7N6MVG3OC4NAJBIDIE_Ea6VqnJSBD6i3UwQ4/edit" title="Site">Google Doc</a></li>
</ul>
  
### Dépendances 
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
