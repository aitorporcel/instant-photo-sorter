# Cellphone Image Classification Project

This project aims to classify a large collection of cellphone images into four specific categories. It's designed for individuals looking to organize and categorize images stored in their devices, particularly those accumulated through apps like WhatsApp. The application utilizes a fine-tuned ResNet50 model for image classification and offers two main functionalities: training your own model with custom data or using the pre-trained model to classify your images.

## Features

- Custom model training with user-provided image data.
- Image classification into four predefined categories using a fine-tuned ResNet50 model.
- A GUI for easy interaction with the application for classification tasks.
- Command-line support for model training and customization.
- TensorBoard integration for training visualization and monitoring.
- Detailed evaluation metrics, including classification reports and confusion matrices.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.8+
- PyTorch 1.8+
- torchvision 0.9+
- PIL (Pillow)
- scikit-learn
- PySide2 (for the GUI)
- TensorBoard (for logging)

### Installation

1. Clone the repository to your local machine:
   ```sh
   git clone https://github.com/your-username/your-project-name.git
   ```
2. Navigate to the project directory and install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Training Your Own Model

To train your own model with custom datasets, follow these steps:

1. **Prepare Your Dataset**: Organize your images into folders named after the categories you wish to classify. For example, if categorizing images into 'Memes', 'Landscapes', 'People', and 'Various', each category should have its own folder containing the relevant images.

2. **Train the Model**: Use the command-line interface to start the training process with your dataset. You can customize training parameters such as the number of epochs, batch size, and train/validation split ratio:

   ```sh
   python main.py --num_epochs 30 --batch_size 32 --train_split 0.8
   ```

3. **Monitor Training Progress**: TensorBoard logs training metrics like loss and accuracy. To view these metrics, run TensorBoard and navigate to the provided URL in your web browser:

   ```sh
   tensorboard --logdir runs/
   ```

4. **Evaluate the Model**: After training, the model is automatically evaluated on the validation and test sets. Results, including the classification report and confusion matrix, are displayed in the terminal.

### Classifying Images with the Pre-trained or Custom Model

To classify your images using the GUI:

1. **Start the Application**: Run `main.py` without any arguments to launch the GUI.

   ```sh
   python main.py
   ```

2. **Select Source and Output Folders**: In the GUI, choose the folder containing the images you wish to classify and specify the output directory where classified images will be stored.

3. **Classify Images**: Click the 'Classify Images' button to start the classification process. Classified images will be copied to subfolders in the output directory, organized by category.

## Project Structure

- `main.py`: The application's entry point.
- `utils.py`: Contains utility functions for training and evaluation.
- `dataset_modelo/`: Default directory for the training dataset.
- `test_data/`: Default directory for the test dataset.
- `models/`: Directory where trained models are saved.

## Contributing

Contributions are welcome! Please feel free to fork the project, make changes, and submit pull requests with new features or improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Add LinkedIn
Project Link: [https://github.com/your-username/your-project-name](https://github.com/your-username/your-project-name)
