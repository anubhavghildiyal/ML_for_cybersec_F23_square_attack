Running Square Attack on Medical MNIST Dataset
==============================================

The Square Attack can be executed against different pre-trained neural network models on the Medical MNIST dataset. Follow these instructions to run the attack using the provided Python script.

How to get the dataset:
-----------------------
You will have to download the medical MNIST dataset from the link below and place it in the same directory as attack.py and name the folder as medical_mnist.

Link:
https://www.kaggle.com/datasets/andrewmvd/medical-mnist

How to Run the Command:
-----------------------

To initiate the Square Attack, use the following command in your terminal:

python attack.py --attack=square_l2 --model=pt_medical_mnist_resnet50 --dataset=medical_mnist --n_ex=1000 --eps=12.75 --p=0.05 --n_iter=10000

Command Options Explained:
--------------------------

--attack: Specifies the type of attack. In this case, 'square_l2' indicates that we are using the L2 version of the Square Attack.

--model: Determines which pre-trained model to use for the attack. Possible values are:
         - pt_medical_mnist_vgg16: Pre-trained VGG16 model on Medical MNIST.
         - pt_medical_mnist_inception: Pre-trained Inception model on Medical MNIST.
         - pt_medical_mnist_resnet50: Pre-trained ResNet50 model on Medical MNIST.

--dataset: Specifies the dataset to be used. 'medical_mnist' is the dataset in this context.

--n_ex: The number of examples to attack. Here, '1000' examples will be attacked.

--eps: The attack perturbation budget (epsilon). '12.75' is the maximum change allowed to the input data.

--p: The percentage of pixels modified in each attack iteration. '0.05' means 5% of pixels will be modified.

--n_iter: The number of iterations for the attack. The script will perform '10000' iterations to find adversarial examples.

Pre-Trained Models:
-------------------

We have included files to train three different neural networks on the Medical MNIST data, which generate the required pre-trained models. These models are essential for performing the attack and have been trained to recognize patterns specific to medical imagery.

Before running the attack, ensure that you have generated and stored the pre-trained model files in the correct directory. The training scripts for each model are as follows:

* For VGG16: train_medical_mnist_vgg16.py
* For Inception: train_medical_mnist_inception.py
* For ResNet50: train_medical_mnist_resnet50.py

The saved models are in the directory medicalMNISTmodels as .pth files.

Python version: 3.6

Each script will output a model file that can be used with the 'python attack.py' command by specifying the corresponding model option.

Note: Make sure you have the necessary Python environment and dependencies set up before running the attack and training scripts.
